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
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphFactory;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.FloatL2Space;
import com.vc.deg.SearchResult;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphSearchBenchmark {

	public static void main(String[] args) throws IOException {
		
		// load query data
		Path siftBaseDir = Paths.get("C:/Data/Feature/SIFT1M/SIFT1M/");
		float[][] queryData = Sift1M.loadQueryData(siftBaseDir);
		int[][] groundtruthData = Sift1M.loadGroundtruthData(siftBaseDir);
		
		// register the feature space needed in the graph
		final AtomicInteger distanceCalculationCount = new AtomicInteger();
		FeatureSpace space = new FloatL2Space(128) { 
			@Override
			public float computeDistance(FeatureVector f1, FeatureVector f2) {
				distanceCalculationCount.incrementAndGet();
				return super.computeDistance(f1, f2);
			}
		};
		FeatureSpace.registerFeatureSpace(space);
		
		// load graph
		final Path graphFile = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.float.deg");
		DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile, float.class.getSimpleName());
		
		// test graph
		final int k = 100;
		final AtomicInteger precisionAtK = new AtomicInteger();
		final AtomicLong elapsedMilliseconds = new AtomicLong();
		IntStream.range(0, queryData.length).parallel().forEach(i -> {
			final float[] query = queryData[i];
			final int[] groundtruth = groundtruthData[i];
			
			// find nearest neighbors
			final long start = System.currentTimeMillis();
			final SearchResult bestList = deg.search(new FloatFeature(query), k);
			final long stop = System.currentTimeMillis();
			elapsedMilliseconds.addAndGet(stop-start);
			
			// check ground truth against
			final IntSet bestIds = HashIntSets.newImmutableSet(c -> bestList.forEach(e -> c.accept(e.getLabel())));
			for(int bestIndex : groundtruth)
				if(bestIds.contains(bestIndex))
					precisionAtK.incrementAndGet();
		});
		
		// final score
		float meanElapsedTime = elapsedMilliseconds.floatValue() / queryData.length;
		float meanPrecisionAtK = precisionAtK.floatValue() / k / queryData.length;
		int meanDistanceCalculationCount = (int)(distanceCalculationCount.get() / queryData.length);
		System.out.printf("avg. precision@%3d: %.5f, avg number of distance calculations: %5d (%6.3fms) \n", k, meanPrecisionAtK, meanDistanceCalculationCount, meanElapsedTime);
	}
}
