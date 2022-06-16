package com.vc.deg.data;

import com.vc.deg.FeatureSpace;
import com.vc.deg.MemoryView;

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
	public float computeDistance(MemoryView f1, MemoryView f2) {
		float result = 0;
		for (int i = 0; i < dims; i++) {
			float diff = f1.readFloat(i) - f2.readFloat(i);
			result += diff*diff;
		}
		return result;
	}
}
