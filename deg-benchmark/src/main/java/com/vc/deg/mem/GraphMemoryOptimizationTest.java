package com.vc.deg.mem;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import com.koloboke.collect.map.IntFloatCursor;
import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphFactory;
import com.vc.deg.anns.GraphSearchBenchmark;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatL2Space;
import com.vc.deg.mem.FLAS1D.MapPlace;
import com.vc.deg.ref.graph.VertexData;

public class GraphMemoryOptimizationTest {

	protected final static Path siftBaseDir = Paths.get("e:\\Data\\Feature\\SIFT1M\\SIFT1M\\");
//	protected final static Path graphFile = Paths.get("e:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg");
//	protected final static Path outputGraphFile = Paths.get("e:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_reorder.deg");
	
	protected final static Path graphFile = Paths.get("e:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0_preOrdered232076720.deg");
	protected final static Path outputGraphFile = Paths.get("e:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0_preOrdered232076720_selfReorder.deg");
	
	
	
	
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		// register the feature space needed in the graph
		FeatureSpace space = new FloatL2Space(128);
		FeatureSpace.registerFeatureSpace(space);
		
//		// initial graph
//		System.out.println("Initial graph vom SIFT1M Base");
//		final DynamicExplorationGraph deg = initialGraph(siftBaseDir, space, 30);
//		System.out.println("ANID of DEG is "+deg.designer().calcAvgNeighborIndexDistance());
//		
//		// compute average feature of the neighbors per vertex
//		System.out.println("Compute average features");
//		final List<float[]> avgVertexFVs = new ArrayList<>(deg.size());
//		for (int i = 0; i < deg.size(); i++) 
//			avgVertexFVs.add(comuteAverageFeature(Arrays.asList(deg.getFeature(i))));
		
		// load graph
		System.out.println("Load graph "+graphFile);
		final DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile, float.class.getSimpleName());
		System.out.println("ANID of DEG is "+deg.designer().calcAvgNeighborIndexDistance());

		// compute average feature of the neighbors per vertex
		System.out.println("Compute average features");
		final List<float[]> avgVertexFVs = new ArrayList<>(deg.size());
		deg.forEachVertex((label, fv) -> {
			final List<FeatureVector> neighborFVs = new ArrayList<>(deg.edgePerVertex());
			deg.forEachNeighbor(label, (int neighborLabel, float weight) -> {
				neighborFVs.add(deg.getFeature(neighborLabel));
			});
//			avgVertexFVs.add(comuteAverageFeature(neighborFVs));
			avgVertexFVs.add(comuteAverageFeature(Arrays.asList(fv)));
		});
		
		// use FLAS to sort the FVs in an optimal 1D order
		System.out.println("Compute optimal memory order");
		final int[] permutation = computeOptimalMemoryOrder(avgVertexFVs, deg);
		
		// rearrange memory in the graph
		System.out.println("Re-arrange vertices in graph");
		final DynamicExplorationGraph memOptDEG = rearrangeVertices(deg, permutation);
		
		// test the new graph
		System.out.println("ANID of rearranged graph is "+memOptDEG.designer().calcAvgNeighborIndexDistance());	
		System.out.println("Store graph "+outputGraphFile);
		memOptDEG.writeToFile(outputGraphFile);
	}

	private static DynamicExplorationGraph initialGraph(Path siftBaseDir, FeatureSpace space, int edgesPerVertex) throws IOException {
		
		final float[][] baseData = Sift1M.loadBaseData(siftBaseDir);
		final int expectedSize = baseData.length;

		com.vc.deg.ref.DynamicExplorationGraph outputGraph = new com.vc.deg.ref.DynamicExplorationGraph(space, expectedSize, edgesPerVertex);
		com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph internalOutputGraph = outputGraph.getInternalGraph();
		
		for (int i = 0; i < baseData.length; i++) 
			internalOutputGraph.addVertex(i, FeatureVector.wrap(baseData[i]));
		
		return outputGraph;
	}

	/**
	 * 
	 * @param deg
	 * @param permutation maps from old id to new id e.g. newId = permutation[oldId]
	 * @return
	 */
	private static DynamicExplorationGraph rearrangeVertices(DynamicExplorationGraph deg, int[] permutation) {
		
		final FeatureSpace space = deg.getFeatureSpace();
		final int expectedSize = deg.size();
		final int edgesPerVertex = deg.edgePerVertex();

		com.vc.deg.ref.DynamicExplorationGraph inputGraph = (com.vc.deg.ref.DynamicExplorationGraph) deg;
		com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph internalInputGraph = inputGraph.getInternalGraph();

		com.vc.deg.ref.DynamicExplorationGraph outputGraph = new com.vc.deg.ref.DynamicExplorationGraph(space, expectedSize, edgesPerVertex);
		com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph internalOutputGraph = outputGraph.getInternalGraph();

		// bring old vertices in new order
		final VertexData[] newOrder = new VertexData[expectedSize];
		for (int oldId = 0; oldId < expectedSize; oldId++) 
			newOrder[permutation[oldId]] = internalInputGraph.getVertexById(oldId);

		// add the old vertices to the new graph
		for (int newId = 0; newId < newOrder.length; newId++) {
			final VertexData oldVertex = newOrder[newId];
			final VertexData newVertex = internalOutputGraph.addVertex(oldVertex.getLabel(), oldVertex.getFeature());
			
			// copy the old edges but remap the old vertex ids to the new ones
			final IntFloatCursor oldEdges = oldVertex.getEdges().cursor();
			while(oldEdges.moveNext())
				newVertex.getEdges().put(permutation[oldEdges.key()], oldEdges.value());
		}
		
		return outputGraph;
	}

	/**
	 * 
	 * @param avgVertexFVs
	 * @return permutation array to map from old id to new id e.g. newId = permutation[oldId]
	 */
	private static int[] computeOptimalMemoryOrder(List<float[]> avgVertexFVs, DynamicExplorationGraph deg) {
		final int count = avgVertexFVs.size();
		final MapPlace[] mapPlaces = new MapPlace[count];
		for (int i = 0; i < mapPlaces.length; i++) 
			mapPlaces[i] = new MapPlace(i, avgVertexFVs.get(i));
		
		FLAS1D.MaxSwapPositions = 30;
		FLAS1D.RadiusDecay = 0.9f;
		
		System.out.println("MaxSwapPositions: "+FLAS1D.MaxSwapPositions+", RadiusDecay: "+FLAS1D.RadiusDecay);
		System.out.printf("Initial Distance %12.0f and FeatureDistance: %12.0f\n", deg.designer().calcAvgNeighborIndexDistance(), computeFeatureDistance(mapPlaces));
		final AtomicLong time = new AtomicLong(System.currentTimeMillis());
		final FLAS1D flas = new FLAS1D(new Random(7), (Integer it, Float rad) -> {
			if(it % 10 == 0) {
				final long duration = System.currentTimeMillis() - time.get();				
				final int[] permutation = new int[count];
				for (int i = 0; i < permutation.length; i++) 
					permutation[mapPlaces[i].getId()] = i;
				final float anid = rearrangeVertices(deg, permutation).designer().calcAvgNeighborIndexDistance();
				final double featureDistance = computeFeatureDistance(mapPlaces);
				System.out.printf("It:%5d Rad: %9.2f Time:%6dms Distance: %12.0f FeatureDistance: %12.0f\n", it, rad, duration, anid, featureDistance);
				time.set(System.currentTimeMillis());
			}
		});
		flas.doSorting(mapPlaces, count, 1);
		
		final int[] permutation = new int[count];
		for (int i = 0; i < permutation.length; i++) 
			permutation[mapPlaces[i].getId()] = i;
		return permutation;
	}
	
	private static double computeFeatureDistance(MapPlace[] data) {		
		float avgDist = 0; 
		for (int i = 0; i < data.length; i++) {
			final float[] entry = data[i].getFloatFeature();
			final float[] nextEntry = data[(i + 1) % data.length].getFloatFeature();			
			avgDist += getL2Distance(entry, nextEntry);
		}
		return avgDist;
	}
	
	private static float getL2Distance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return (float) Math.sqrt(dist);
	}

	private static float[] comuteAverageFeature(List<FeatureVector> neighborFVs) {
		
		final int dims = neighborFVs.get(0).dims();
		final float[] fv = new float[dims];
		for (FeatureVector neighborFV : neighborFVs) 
			for (int i = 0; i < fv.length; i++) 
				fv[i] += neighborFV.readFloat(i * Float.BYTES);

		final int count = neighborFVs.size();
		for (int i = 0; i < fv.length; i++) 
			fv[i] /= count;
		
		return fv;
	}
}