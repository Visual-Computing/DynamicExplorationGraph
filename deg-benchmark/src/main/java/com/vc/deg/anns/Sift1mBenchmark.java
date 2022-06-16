package com.vc.deg.anns;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.data.FeatureSpace;
import com.vc.deg.data.StaticFeatureRepository;

public class Sift1mBenchmark {

	public static void main(String[] args) {
		int edgesPerNode = 4;
		float[][] data = loadSiftData();
		FeatureSpace<float[]> space = (f1, f2) -> 0;
		StaticFeatureRepository<float[]> repo = new StaticFeatureRepository<>(space, data);		
		DynamicExplorationGraph<float[]> dng = new DynamicExplorationGraph<>(repo, edgesPerNode);
	}

	private static float[][] loadSiftData() {
		// TODO Auto-generated method stub
		return null;
	}

}
