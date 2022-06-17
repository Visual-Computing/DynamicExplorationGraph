package com.vc.deg.data;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;

public class FloatL2Space implements FeatureSpace {

	protected int dims;
	
	public FloatL2Space(int dims) {
		this.dims = dims;
	}
	
	@Override
	public int size() {
		return dims*4;
	}
	
	@Override
	public float computeDistance(FeatureVector f1, FeatureVector f2) {
		float result = 0;
		for (int i = 0; i < dims; i++) {
			float diff = f1.readFloat(i) - f2.readFloat(i);
			result += diff*diff;
		}
		return result;
	}
}
