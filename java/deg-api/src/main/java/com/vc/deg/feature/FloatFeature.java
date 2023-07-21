package com.vc.deg.feature;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an float-array
 * 
 * @author Nico Hezel
 */
public class FloatFeature implements FeatureVector {
	
	protected final float[] feature;
	
	public FloatFeature(float[] feature) {
		this.feature = feature;
	}
	
	public float get(int index) {
		return feature[index];
	}
	
	@Override
	public int dims() {
		return feature.length;
	}
	
	@Override
	public Class<?> getComponentType() {
		return float.class;
	}

	@Override
	public int size() {
		return feature.length * Float.BYTES;
	}

	@Override
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " does not support readLong");
	}

	@Override
	public float readFloat(int byteOffset) {
		return feature[byteOffset >> 2];
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " does not support readDouble");
	}

//	@Override
//	public byte[] toBytes() {
//		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
//		for (float value : feature) 
//			bb.putFloat(value);
//		return bb.array();
//	}
	
	@Override
	public FeatureVector copy() {
		return new FloatFeature(Arrays.copyOf(feature, feature.length));
	}
	
	
	@Override
	public void writeObject(DataOutput out) throws IOException {
		for (float d : feature) 
			out.writeFloat(d);
	}

	@Override
	public void readObject(DataInput in) throws IOException {
		for (int i = 0; i < feature.length; i++) 
			feature[i] = in.readFloat();
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(FloatFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}