package com.vc.deg.ref.feature;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;

public class L2FloatSpace implements FeatureSpace {
	
	protected int dims;
	
	public L2FloatSpace(int dims) {
		this.dims = dims;
	}

	@Override
	public String getComponentType() {
		return float.class.getSimpleName();
	}
	
	@Override
	public int dims() {
		return dims;
	}
	
	@Override
	public int featureSize() {
		return Float.BYTES * dims();
	}
	
	@Override
	public float computeDistance(FeatureVector f1, FeatureVector f2) {
		final int byteSize = f1.size();
		
		float sum = 0;
		for (int i = 0; i < byteSize; i+=4) {
			float diff = f1.readFloat(i) - f2.readFloat(i);
			sum += diff*diff;
		}
		
		return sum;
	}

	@Override
	public int metric() {
		return Metric.L2.getId();
	}

	@Override
	public boolean isNative() {
		return false;
	}
}
