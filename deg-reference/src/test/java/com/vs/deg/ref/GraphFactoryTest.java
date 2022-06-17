package com.vs.deg.ref;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.IntFeature;
import com.vc.deg.ref.DynamicExplorationGraph;
import com.vc.deg.ref.GraphFactory;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;

public class GraphFactoryTest {

	public static void main(String[] args) {

		// feature vector size and distance calculation
		// the feature space 
		FeatureSpace space = new FeatureSpace() {
			
			@Override
			public int size() {
				return 4;
			}
			
			@Override
			public float computeDistance(FeatureVector f1, FeatureVector f2) {
				return Math.abs(f1.readInt(0) - f2.readInt(0));
			}
		};
		
		// construct a empty graph
		GraphFactory factory = new GraphFactory();
		DynamicExplorationGraph graph = factory.newGraph(space, 4);

		// create example data
		final int datasetSize = 100;
		final Data[] dataset = new Data[datasetSize];
		for (int i = 0; i < dataset.length; i++) 
			dataset[i] = new Data(i, new int[] {i});
		
		// extend the graph
		EvenRegularGraphDesigner designer = graph.designer();
		for (Data data : dataset) {
			designer.add(data.label, data.feature);
		}
		
	}

	
	/**
	 * Wrapper for test data
	 * 
	 * @author Nico Heze
	 */
	protected static class Data {
		protected final int label;
		protected final FeatureVector feature;
		
		public Data(int label, int[] feature) {
			this.label = label;
			this.feature = new IntFeature(feature);
		}
	}
}