package com.vc.deg.anns;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.GraphFactory;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.FloatL2Space;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphSearchBenchmark {

	public final static Path siftBaseDir = Paths.get("e:/Data/Feature/SIFT1M/SIFT1M/");
	public final static Path graphFile   = Paths.get("e:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg");

	public static void main(String[] args) throws IOException {
		testGraph(graphFile, siftBaseDir);
	}
	
	public static void testGraph(Path graphFile, Path siftBaseDir) throws IOException {
		
		// register the feature space needed in the graph
		FeatureSpace.registerFeatureSpace(new FloatL2Space(128));
		
		// load graph
		DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile, float.class.getSimpleName());
		
		// test the graph
		testGraph(deg, siftBaseDir);
	}
	
	public static void testGraph(DynamicExplorationGraph deg, Path siftBaseDir) throws IOException {

		// load query data
		float[][] queryData = Sift1M.loadQueryData(siftBaseDir);
		int[][] groundtruthData = Sift1M.loadGroundtruthData(siftBaseDir);
		
		// test graph
		final int k = 100;		
		for(float eps : new float[]{0.01f, 0.05f, 0.1f, 0.12f}) {	
			final AtomicInteger precisionAtK = new AtomicInteger();
			final AtomicLong elapsedMilliseconds = new AtomicLong();
			IntStream.range(0, queryData.length).parallel().forEach(i -> {
				final float[] query = queryData[i];
				final int[] groundtruth = groundtruthData[i];
				
				// find nearest neighbors
				final long start = System.currentTimeMillis();
				final int[] bestList = deg.search(new FloatFeature(query), k, eps);
				final long stop = System.currentTimeMillis();
				elapsedMilliseconds.addAndGet(stop-start);
				
				// check ground truth against
				final IntSet bestIds = HashIntSets.newImmutableSet(bestList);
				for(int bestIndex : groundtruth)
					if(bestIds.contains(bestIndex))
						precisionAtK.incrementAndGet();
			});
			
			// final score
			float meanElapsedTime = elapsedMilliseconds.floatValue() / queryData.length;
			float meanPrecisionAtK = precisionAtK.floatValue() / k / queryData.length;
			int queriesPerSecond = (int)(1000 / meanElapsedTime);
			System.out.printf("avg. precision@%3d: %.5f at %4d queries/sec (avg %6.3fms/query) \n", k, meanPrecisionAtK, queriesPerSecond, meanElapsedTime);
		}
	}
}
