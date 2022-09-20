package com.vc.deg.feature;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;

public class FloatL2Space implements FeatureSpace {

	protected int dims;
	
	public FloatL2Space(int dims) {
		this.dims = dims;
	}
	
	@Override
	public int featureSize() {
		return Float.BYTES * dims();
	}
	
	@Override
	public int dims() {
		return dims;
	}
	
	@Override
	public String getComponentType() {
		return float.class.getSimpleName();
	}


	@Override
	public int metric() {
		return Metric.L2.getId();
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
	
	@Override
	public float computeDistance(FeatureVector f1, FeatureVector f2) {
		final int byteSize = f1.size();
		
		float result = 0;
		for (int i = 0; i < byteSize; i+=4) {
			float diff = f1.readFloat(i) - f2.readFloat(i);
			result += diff*diff;
		}
		
		return result;
	}
}
