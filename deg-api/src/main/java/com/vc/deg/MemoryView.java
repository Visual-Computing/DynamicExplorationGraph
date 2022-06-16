package com.vc.deg;

/**
 * A piece of memory
 * 
 * @author Nico Hezel
 */
public interface MemoryView {
	
	/**
	 * Size in bytes of this slice
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Read a boolean (single bit) from the index 
	 * 
	 * @param index
	 * @return
	 */
	public boolean readBoolean(long index);
	
	/**
	 * Read a byte from the index
	 * 
	 * @param index
	 * @return
	 */
	public byte readByte(long index);
	
	/**
	 * Read a short from the index
	 * 
	 * @param index
	 * @return
	 */
	public short readShort(long index);
	
	/**
	 * Read a int from the index
	 * 
	 * @param index
	 * @return
	 */
	public int readInt(long index);
	
	/**
	 * Read a long from the index
	 * 
	 * @param index
	 * @return
	 */
	public long readLong(long index);
	
	/**
	 * Read a float from the index
	 * 
	 * @param index
	 * @return
	 */
	public float readFloat(long index);
	
	/**
	 * Read a double from the index
	 * 
	 * @param index
	 * @return
	 */
	public double readDouble(long index);
	
	/**
	 * Copy some internal bytes to the other memory slice
	 * 
	 * @param offset
	 * @param other
	 * @param otherOffset
	 * @param length in bytes
	 */
	public void copyTo(long offset, MemoryView other, long otherOffset, long length);
	
	/**
	 * Copy some bytes from the other memory slice to the internal bytes
	 * 
	 * @param offset
	 * @param other
	 * @param otherOffset
	 * @param length in bytes
	 */
	public void copyFrom(long offset, MemoryView other, long otherOffset, long length);
}