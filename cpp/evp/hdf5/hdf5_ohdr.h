#pragma once

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {
namespace detail {

// ============================================================================
// Message payload processor (shared between OHdr v1 and v2)
// ============================================================================
inline void apply_msg(std::ifstream& f, uint16_t mtype, uint64_t mdata_abs,
                      uint64_t base, OhdrInfo& info)
{
    if (mtype == 1) {
        // ---- Dataspace message ----
        fseek(f, mdata_abs);
        uint8_t ds_ver = frd8(f);
        uint8_t rank = frd8(f);
        uint8_t ds_flags = frd8(f);
        if (ds_ver == 1) {
            fskip(f, 5); // v1: 5 reserved bytes after flags
        } else {
            fskip(f, 1); // v2+: 1 type byte (H5S_SIMPLE etc.) after flags
        }
        if (rank >= 1) info.dim0 = frd64(f);
        if (rank >= 2) info.dim1 = frd64(f);
        // Skip max dims if present (flags & H5S_VALID_MAX = 0x01)
        if (ds_flags & 0x01) {
            if (rank >= 1) fskip(f, 8); // max[0]
            if (rank >= 2) fskip(f, 8); // max[1]
        }
    } else if (mtype == 3) {
        // ---- Datatype message ----
        // class_and_ver(1) + bitfields(3) + size(4)
        fseek(f, mdata_abs);
        fskip(f, 4);
        info.elem_size = frd32(f);
    } else if (mtype == 8) {
        // ---- Data Layout message ----
        fseek(f, mdata_abs);
        uint8_t lver = frd8(f);
        if (lver == 1 || lver == 2) {
            // v1/v2: dimensionality(1) layout_class(1) reserved(5) data_addr(8)
            fskip(f, 1);        // dimensionality
            uint8_t lcls = frd8(f);
            fskip(f, 5);        // reserved
            uint64_t ar = frd64(f);
            if (lcls == 1 && ar != UNDEF64) info.data_abs = base + ar;
            // data_length not in v1/v2 layout — compute from dataspace later
        } else if (lver == 3 || lver == 4) {
            // v3/v4: layout_class(1) then if contiguous: addr(8) + len(8)
            uint8_t lcls = frd8(f);
            if (lcls == 1) {   // contiguous
                uint64_t ar = frd64(f);
                uint64_t dln = frd64(f);
                if (ar != UNDEF64) { info.data_abs = base + ar; info.data_len = dln; }
            }
        }
    } else if (mtype == 17) {
        // ---- Symbol Table message → group ----
        fseek(f, mdata_abs);
        uint64_t br = frd64(f), hr = frd64(f);
        info.is_group = true;
        info.group_btree_abs = (br != UNDEF64) ? base + br : UNDEF64;
        info.group_heap_abs = (hr != UNDEF64) ? base + hr : UNDEF64;
    } else if (mtype == 6) {
        // ---- Link Message ----
        fseek(f, mdata_abs);
        uint8_t lver = frd8(f);
        if (lver == 1) {
            uint8_t lflags = frd8(f);
            uint8_t link_type = 0;
            if (lflags & 0x08) {
                link_type = frd8(f);
            }
            if (lflags & 0x04) {
                fskip(f, 8); // creation order
            }
            if (lflags & 0x10) {
                fskip(f, 1); // character set
            }
            // link name length: 1u << (lflags & 0x03) bytes
            uint64_t name_len = 0;
            uint8_t name_len_sz = 1u << (lflags & 0x03);
            if (name_len_sz == 1) name_len = frd8(f);
            else if (name_len_sz == 2) name_len = frd16(f);
            else if (name_len_sz == 4) name_len = frd32(f);
            else name_len = frd64(f);

            // Read name
            std::string name(name_len, '\0');
            f.read(&name[0], (std::streamsize)name_len);

            // If hard link (type 0), read address
            if (link_type == 0) {
                uint64_t obj_rel = frd64(f);
                if (obj_rel != UNDEF64) {
                    info.links.push_back({name, base + obj_rel});
                    // Having links makes this object a group
                    info.is_group = true;
                }
            }
        }
    }
}

// ============================================================================
// Object Header v1
// First byte == 1, followed by fixed header then 8-byte aligned messages.
// ============================================================================
inline void parse_ohdr_messages_v1(std::ifstream& f, uint64_t block_abs, uint32_t block_size, uint64_t base, OhdrInfo& info, uint16_t& msgs_left) {
    uint32_t consumed = 0;
    while (consumed < block_size && msgs_left > 0) {
        fseek(f, block_abs + consumed);
        uint16_t mtype = frd16(f);
        uint16_t msize = frd16(f);
        fskip(f, 1 + 3);              // flags + reserved
        uint64_t mdata_abs = block_abs + consumed + 8;

        if (mtype == 16) {
            // Continuation message: read offset and length (using 8-byte offsets/lengths)
            fseek(f, mdata_abs);
            uint64_t cont_off = frd64(f);
            uint64_t cont_len = frd64(f);
            msgs_left--;
            // Recursively parse the continuation block
            parse_ohdr_messages_v1(f, base + cont_off, (uint32_t)cont_len, base, info, msgs_left);
        } else {
            apply_msg(f, mtype, mdata_abs, base, info);
            msgs_left--;
        }

        uint32_t padded = (8u + msize + 7u) & ~7u;
        consumed += padded;
    }
}

inline OhdrInfo parse_ohdr_v1(std::ifstream& f, uint64_t ohdr_abs, uint64_t base) {
    fseek(f, ohdr_abs);
    uint8_t ver = frd8(f);
    if (ver != 1) throw std::runtime_error(
        "hdf5_reader: ohdr_v1 bad version=" + std::to_string(ver) + " at " + std::to_string(ohdr_abs));
    fskip(f, 1);                    // reserved
    uint16_t n_msgs = frd16(f);
    frd32(f);                       // obj_ref_count
    uint32_t hdr_sz = frd32(f);
    fskip(f, 4);                    // reserved
    // messages start at offset 16 from ohdr_abs

    OhdrInfo info;
    uint16_t msgs_left = n_msgs;
    parse_ohdr_messages_v1(f, ohdr_abs + 16, hdr_sz, base, info, msgs_left);

    // Compute data_len if not set by layout message
    if (info.data_len == 0 && info.elem_size > 0) {
        uint64_t nelems = info.dim0;
        if (info.dim1 > 0) nelems *= info.dim1;
        info.data_len = nelems * info.elem_size;
    }

    return info;
}

// ============================================================================
// Object Header v2
// Starts with "OHDR" signature, variable-size chunk0, messages with optional
// creation-order field.
// ============================================================================
inline OhdrInfo parse_ohdr_v2(std::ifstream& f, uint64_t ohdr_abs, uint64_t base) {
    fseek(f, ohdr_abs);
    fskip(f, 4);                 // "OHDR" signature
    uint8_t ver = frd8(f);
    if (ver != 2) throw std::runtime_error(
        "hdf5_reader: ohdr_v2 bad version=" + std::to_string(ver) + " at " + std::to_string(ohdr_abs));
    uint8_t flags = frd8(f);

    // Optional timestamps: 4 x uint32 if bit 5 of flags set
    if ((flags >> 5) & 1) fskip(f, 16);
    // Optional phase-change values: 2 x uint16 if bit 4 set
    if ((flags >> 4) & 1) fskip(f, 4);

    // chunk#0 size field width: bits 1:0 of flags → 1, 2, 4, or 8 bytes
    uint8_t chunk_bytes = 1u << (flags & 0x3);
    uint64_t chunk0_size = 0;
    { uint8_t buf[8] = {}; f.read((char*)buf, chunk_bytes); memcpy(&chunk0_size, buf, chunk_bytes); }

    // Whether creation-order field is present in each message header
    bool has_creation_order = (flags >> 2) & 1;

    // v2 message header: type(1) + size(2) + flags(1) + [cre_order(2)] = 4 or 6 bytes
    uint64_t msg_hdr_size = 4u + (has_creation_order ? 2u : 0u);

    OhdrInfo info;
    uint64_t consumed = 0;
    while (consumed < chunk0_size) {
        uint8_t  mtype      = frd8(f);
        uint16_t msize      = frd16(f);
        frd8(f);             // message flags
        if (has_creation_order) fskip(f, 2);
        uint64_t mdata_abs = (uint64_t)f.tellg();

        if (msize > 0) apply_msg(f, mtype, mdata_abs, base, info);

        consumed += msg_hdr_size + msize;
        fseek(f, mdata_abs + msize);
    }

    // Skip trailing CRC32 checksum (4 bytes) — not needed for parsing
    fskip(f, 4);

    // Compute data_len if missing
    if (info.data_len == 0 && info.elem_size > 0) {
        uint64_t nelems = info.dim0;
        if (info.dim1 > 0) nelems *= info.dim1;
        info.data_len = nelems * info.elem_size;
    }
    return info;
}

// ============================================================================
// Dispatch: detect v1 vs v2 and parse accordingly
// ============================================================================
inline OhdrInfo parse_ohdr(std::ifstream& f, uint64_t ohdr_abs, uint64_t base) {
    fseek(f, ohdr_abs);
    uint8_t first4[4]; f.read((char*)first4, 4);
    if (memcmp(first4, SIG_OHDR, 4) == 0) {
        return parse_ohdr_v2(f, ohdr_abs, base);
    }
    if (first4[0] != 1) throw std::runtime_error(
        "hdf5_reader: unknown OHdr format (first byte=" + std::to_string(first4[0])
        + ") at " + std::to_string(ohdr_abs));
    return parse_ohdr_v1(f, ohdr_abs, base);
}

} // namespace detail
} // namespace hdf5_reader
