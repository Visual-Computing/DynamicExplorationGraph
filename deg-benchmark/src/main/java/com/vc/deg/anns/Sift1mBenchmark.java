package com.vc.deg.anns;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.GraphDesigner;
import com.vc.deg.GraphFactory;
import com.vc.deg.FeatureVector;
import com.vc.deg.data.Describable;
import com.vc.deg.data.FloatL2Space;
import com.vc.deg.data.Identifiable;

public class Sift1mBenchmark {

	public static void main(String[] args) {
		int edgesPerNode = 4;
		SiftFeature[] data = loadSiftData();
		FeatureSpace space = new FloatL2Space(128);
		DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().newGraph(space, edgesPerNode);
		GraphDesigner builder = deg.designer();
		
		for (int i = 0; i < data.length; i++) {
			final SiftFeature entry = data[i];
			builder.add(entry.getId(), entry.getFeature());
		}
	}

	private static SiftFeature[] loadSiftData() {
		// TODO Auto-generated method stub
		return null;
	}

	public static class SiftFeature implements Identifiable, Describable {
		protected int id;
		protected FeatureVector feature;
		
		public SiftFeature(int id, FeatureVector feature) {
			this.id = id;
			this.feature = feature;
		}
		
		@Override
		public int getId() {
			return id;
		}
		
		@Override
		public FeatureVector getFeature() {
			return feature;
		}
	}
}
