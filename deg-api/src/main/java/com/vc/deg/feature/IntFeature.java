package com.vc.deg.feature;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an int-array
 * 
 * @author Nico Hezel
 */
public class IntFeature implements FeatureVector {
	
	protected final int[] feature;
	
	public IntFeature(int[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length * Integer.BYTES;
	}

	@Override
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		return feature[byteOffset >> 2];
	}

	@Override
	public long readLong(int byteOffset) {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " does not support readLong");
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (int value : feature) 
			bb.putInt(value);
		return bb.array();
	}
	
	@Override
	public FeatureVector copy() {
		return new IntFeature(Arrays.copyOf(feature, feature.length));
	}
	
	@Override
	public void writeObject(DataOutputStream out) throws IOException {
		for (int d : feature) 
			out.writeInt(d);
	}

	@Override
	public void readObject(DataInputStream in) throws IOException {
		for (int i = 0; i < feature.length; i++) 
			feature[i] = in.readInt();
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(IntFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}