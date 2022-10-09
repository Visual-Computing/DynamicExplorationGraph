package com.vc.deg;

import java.io.DataInput;
import java.io.IOException;

/**
 * TODO a little to much, it can transform from and feature to any feature
 *      But we need to transform from any feature to float array
 * 
 * @author Nico Hezel
 *
 */
public interface FeatureTransformer {

	public FeatureVector transform(FeatureVector fv);
	
	
	public static class Byte8LongToFloatTransformer implements FeatureTransformer {
		
		protected final int dims;
		protected final FeatureFactory factory;
		
		public Byte8LongToFloatTransformer(int dims, FeatureFactory factory) {
			this.dims = dims;
			this.factory = factory;
			
			if(factory.getComponentType() != float.class)
				throw new RuntimeException("The provided feature factory does not produce float features but instead "+factory.getComponentType()+" features.");
		}

		@Override
		public FeatureVector transform(FeatureVector fv) {
			
			final long[] longs = new long[8];
			for (int i = 0; i < longs.length; i++) 
				longs[i] = fv.readLong(i * Long.BYTES);
			
			final int size = 64;
			final int bitCount = 8;
			final byte[] bytes = new byte[size];
			for (int byteIndex = 0; byteIndex < size; byteIndex++) {
				int out = 0;
				for (int bitIndex = 0; bitIndex < bitCount; bitIndex++)
					out |= (longs[bitIndex] >> byteIndex & 1) << bitIndex;			
				bytes[byteIndex] = (byte) (out - 128);
			}
			
			DataInput input = new LazyDataInput() {
				
				protected int pos = 0;
				protected final byte[] byteArray = bytes;
				
				@Override
				public float readFloat() throws IOException {
					final float value = byteArray[pos];
					pos++;
					return value;
				}
			};
					
			try {
				return factory.read(input);
			} catch (IOException e) {
				e.printStackTrace();
			}
			return null;
		}		
	}
	
	public static class ByteToFloatTransformer implements FeatureTransformer {
		
		protected final int dims;
		protected final FeatureFactory factory;
		
		public ByteToFloatTransformer(int dims, FeatureFactory factory) {
			this.dims = dims;
			this.factory = factory;
			
			if(factory.getComponentType() != float.class)
				throw new RuntimeException("The provided feature factory does not produce float features but instead "+factory.getComponentType()+" features.");
		}

		@Override
		public FeatureVector transform(FeatureVector fv) {
			DataInput input = new LazyDataInput() {
				
				protected int pos = 0;
				protected final FeatureVector feature = fv;
				
				@Override
				public float readFloat() throws IOException {
					final float value = feature.readFloat(pos * Float.BYTES);
					pos++;
					return value;
				}
			};
					
			try {
				return factory.read(input);
			} catch (IOException e) {
				e.printStackTrace();
			}
			return null;
		}		
	}
	
	public static class LazyDataInput implements DataInput {

		@Override
		public void readFully(byte[] b) throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public void readFully(byte[] b, int off, int len) throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public int skipBytes(int n) throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public boolean readBoolean() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public byte readByte() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public int readUnsignedByte() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public short readShort() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public int readUnsignedShort() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public char readChar() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public int readInt() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public long readLong() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public float readFloat() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public double readDouble() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public String readLine() throws IOException {
			throw new UnsupportedOperationException();
		}

		@Override
		public String readUTF() throws IOException {
			throw new UnsupportedOperationException();
		}		
	}
}
