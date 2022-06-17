package com.vc.deg.feature;

import java.util.BitSet;

import com.vc.deg.FeatureVector;

/**
 * Wraps an bit-array
 * 
 * @author Nico Hezel
 */
public class BooleanFeature implements FeatureVector {
	
	protected final BitSet bits;
	
	public BooleanFeature(byte[] feature) {
		this.bits = BitSet.valueOf(feature);
	}
	
	public BooleanFeature(long[] feature) {
		this.bits = BitSet.valueOf(feature);
	}

	@Override
	public int size() {
		return (bits.length()+7)/8;
	}

	@Override
	public boolean readBoolean(long index) {
		return bits.get((int)index);
	}

	@Override
	public byte readByte(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readByte");
	}

	@Override
	public short readShort(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readShort");
	}

	@Override
	public int readInt(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readInt");
	}

	@Override
	public long readLong(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readLong");
	}

	@Override
	public float readFloat(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readFloat");
	}

	@Override
	public double readDouble(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		return bits.toByteArray();
	}
}