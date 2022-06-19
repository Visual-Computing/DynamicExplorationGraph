package com.vs.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.IntFeature;
import com.vc.deg.ref.DynamicExplorationGraph;
import com.vc.deg.ref.GraphFactory;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.feature.L2FloatSpace;
import com.vc.deg.ref.graph.EvenRegularWeightedUndirectedGraph;

public class GraphFactoryTest {

	public static void main(String[] args) throws ClassNotFoundException, IOException {
		loadTest();
		System.out.println("Finished");
	}
	
	public static void loadTest() throws ClassNotFoundException, IOException {
		final Path graphFile = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.float.deg");
		
		FeatureSpace siftL2Space = new L2FloatSpace(128);
		

		// loading an existing sift1m deg graph
		FeatureSpace.registerFeatureSpace(siftL2Space);
//		GraphFactory factory = new GraphFactory();
//		DynamicExplorationGraph graph = factory.loadGraph(graphFile, float.class.getSimpleName());
		
		long start = System.currentTimeMillis();
		EvenRegularWeightedUndirectedGraph graph = EvenRegularWeightedUndirectedGraph.readFromFile(graphFile, float.class.getSimpleName());
		long end = System.currentTimeMillis();
		
		System.out.println("loading a graph with "+graph.getNodeIds().size()+" nodes took "+(end-start)+"ms");
		
		// search with the last node in the graph
	}
	
	public static void createTest() {
		// feature vector size and distance calculation
		FeatureSpace space = new L2FloatSpace(128);
		
		// construct a empty graph
		GraphFactory factory = new GraphFactory();
		DynamicExplorationGraph graph = factory.newGraph(space, 4);
		
		// create example data
		final int datasetSize = 100;
		final Data[] dataset = new Data[datasetSize];
		for (int i = 0; i < dataset.length; i++) 
			dataset[i] = new Data(i, new float[] {i});
		
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
		
		public Data(int label, float[] feature) {
			this.label = label;
			this.feature = new FloatFeature(feature);
		}
	}
}