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
import com.vc.deg.feature.FloatL2Space;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphExploreBenchmark {

	protected final static Path siftBaseDir = Paths.get("C:/Data/Feature/SIFT1M/SIFT1M/");
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.float.deg");
	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.rng_optimized.deg");

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
		int[][] queryData = Sift1M.loadExploreQueryData(siftBaseDir);
		int[][] groundtruthData = Sift1M.loadExploreGroundtruthData(siftBaseDir);
		
		// test graph
		final int k = 1000;		
		for (int t = 1; t < 10; t++) {	
			int maxDistanceCalcCount = k*t;
			final AtomicInteger precisionAtK = new AtomicInteger();
			final AtomicLong elapsedMilliseconds = new AtomicLong();
			IntStream.range(0, queryData.length).parallel().forEach(i -> {
				final int entryVertex = queryData[i][0];
				final int[] groundtruth = groundtruthData[i];
				
				// find nearest neighbors
				final long start = System.currentTimeMillis();
				final int[] bestList = deg.explore(entryVertex, k, maxDistanceCalcCount);
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
			System.out.printf("avg. precision@%3d: %.5f at %4d queries/sec (avg %6.3fms/query) with maxDistanceCalcCount: %6d\n", k, meanPrecisionAtK, queriesPerSecond, meanElapsedTime, maxDistanceCalcCount);
		}
	}
}
