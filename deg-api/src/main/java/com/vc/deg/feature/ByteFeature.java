package com.vc.deg.feature;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an byte-array
 * 
 * @author Nico Hezel
 */
public class ByteFeature implements FeatureVector {
	
	protected final byte[] feature;
	
	public ByteFeature(byte[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length;
	}

	@Override
	public byte readByte(int byteOffset) {
		return feature[byteOffset];
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " does not support readLong");
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		return Arrays.copyOf(feature, feature.length);
	}
	
	@Override
	public FeatureVector copy() {
		return new ByteFeature(toBytes());
	}

	@Override
	public void writeObject(DataOutputStream out) throws IOException {
		out.write(feature);		
	}

	@Override
	public void readObject(DataInputStream in) throws IOException {
		in.read(feature);
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(ByteFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}