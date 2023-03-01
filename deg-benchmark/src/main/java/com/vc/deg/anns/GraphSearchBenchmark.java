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
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphSearchBenchmark {

	protected final static Path siftBaseDir = Paths.get("C:/Data/Feature/SIFT1M/SIFT1M/");
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.float.deg");
	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd3+2_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.deg");

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
	
	/** orig
avg. precision@100: 0.94675 at  261 queries/sec (avg  3.819ms/query), avg cals/search 679.494 approaching 1524.663 exploring, avg hops/search 28.048 approaching 90.660 exploring
avg. precision@100: 0.97622 at  206 queries/sec (avg  4.839ms/query), avg cals/search 679.494 approaching 2352.749 exploring, avg hops/search 28.048 approaching 144.047 exploring
avg. precision@100: 0.99322 at  147 queries/sec (avg  6.795ms/query), avg cals/search 679.494 approaching 3855.797 exploring, avg hops/search 28.048 approaching 248.320 exploring
avg. precision@100: 0.99615 at  131 queries/sec (avg  7.589ms/query), avg cals/search 679.494 approaching 4629.897 exploring, avg hops/search 28.048 approaching 305.481 exploring
	
	 * if(isApproaching == false && nextVertex.getVertex().getEdges().get(neighborId) > radius && checkRNG(nextVertex.getVertex().getId(), neighborId, nextVertex.getVertex().getEdges().get(neighborId))) continue;
avg. precision@100: 0.94430 at  206 queries/sec (avg  4.833ms/query), avg cals/search 679.494 approaching 1391.015 exploring, avg hops/search 28.048 approaching 90.648 exploring
avg. precision@100: 0.97384 at  176 queries/sec (avg  5.674ms/query), avg cals/search 679.494 approaching 2135.683 exploring, avg hops/search 28.048 approaching 143.844 exploring
avg. precision@100: 0.99157 at  105 queries/sec (avg  9.447ms/query), avg cals/search 679.494 approaching 3480.492 exploring, avg hops/search 28.048 approaching 247.713 exploring
avg. precision@100: 0.99479 at   93 queries/sec (avg 10.676ms/query), avg cals/search 679.494 approaching 4171.042 exploring, avg hops/search 28.048 approaching 304.614 exploring
avg. precision@100: 0.99540 at   75 queries/sec (avg 13.325ms/query), avg cals/search 679.494 approaching 4358.940 exploring, avg hops/search 28.048 approaching 320.481 exploring
avg. precision@100: 0.99589 at   84 queries/sec (avg 11.802ms/query), avg cals/search 679.494 approaching 4552.246 exploring, avg hops/search 28.048 approaching 337.003 exploring
avg. precision@100: 0.99636 at   79 queries/sec (avg 12.541ms/query), avg cals/search 679.494 approaching 4752.679 exploring, avg hops/search 28.048 approaching 354.329 exploring
avg. precision@100: 0.99676 at   78 queries/sec (avg 12.797ms/query), avg cals/search 679.494 approaching 4959.666 exploring, avg hops/search 28.048 approaching 372.390 exploring


	
	     
     * if(isApproaching == false && nextVertex.getVertex().getEdges().get(neighborId) > radius) continue;
avg. precision@100: 0.94322 at  205 queries/sec (avg  4.858ms/query), avg cals/search 679.494 approaching 1346.936 exploring, avg hops/search 28.048 approaching 90.626 exploring
avg. precision@100: 0.97279 at  257 queries/sec (avg  3.891ms/query), avg cals/search 679.494 approaching 2067.792 exploring, avg hops/search 28.048 approaching 143.758 exploring
avg. precision@100: 0.99064 at  144 queries/sec (avg  6.909ms/query), avg cals/search 679.494 approaching 3369.161 exploring, avg hops/search 28.048 approaching 247.467 exploring
avg. precision@100: 0.99163 at  135 queries/sec (avg  7.381ms/query), avg cals/search 679.494 approaching 3527.124 exploring, avg hops/search 28.048 approaching 260.690 exploring
avg. precision@100: 0.99250 at  148 queries/sec (avg  6.727ms/query), avg cals/search 679.494 approaching 3691.620 exploring, avg hops/search 28.048 approaching 274.604 exploring
avg. precision@100: 0.99323 at  148 queries/sec (avg  6.725ms/query), avg cals/search 679.494 approaching 3860.704 exploring, avg hops/search 28.048 approaching 289.055 exploring
avg. precision@100: 0.99391 at  131 queries/sec (avg  7.607ms/query), avg cals/search 679.494 approaching 4036.975 exploring, avg hops/search 28.048 approaching 304.249 exploring
	
	 * if(isApproaching && nextVertex.getVertex().getEdges().get(neighborId) > radius * (1 + eps)) continue;
avg. precision@100: 0.94656 at  255 queries/sec (avg  3.917ms/query), avg cals/search 657.955 approaching 1529.432 exploring, avg hops/search 27.878 approaching 90.837 exploring
avg. precision@100: 0.97616 at  238 queries/sec (avg  4.192ms/query), avg cals/search 662.392 approaching 2356.240 exploring, avg hops/search 27.948 approaching 144.146 exploring
avg. precision@100: 0.99321 at  150 queries/sec (avg  6.650ms/query), avg cals/search 666.193 approaching 3858.555 exploring, avg hops/search 27.991 approaching 248.374 exploring
avg. precision@100: 0.99614 at  138 queries/sec (avg  7.244ms/query), avg cals/search 667.427 approaching 4632.340 exploring, avg hops/search 28.007 approaching 305.515 exploring

	 * if(isApproaching == false && nextVertex.getVertex().getEdges().get(neighborId) > radius * (1 + eps)) continue;
avg. precision@100: 0.94364 at  257 queries/sec (avg  3.879ms/query), avg cals/search 679.494 approaching 1355.520 exploring, avg hops/search 28.048 approaching 90.632 exploring
avg. precision@100: 0.97440 at  244 queries/sec (avg  4.094ms/query), avg cals/search 679.494 approaching 2128.563 exploring, avg hops/search 28.048 approaching 143.884 exploring
avg. precision@100: 0.99249 at  154 queries/sec (avg  6.484ms/query), avg cals/search 679.494 approaching 3547.305 exploring, avg hops/search 28.048 approaching 248.041 exploring
avg. precision@100: 0.99568 at  138 queries/sec (avg  7.212ms/query), avg cals/search 679.494 approaching 4282.429 exploring, avg hops/search 28.048 approaching 305.159 exploring

	 * if(nextVertex.getVertex().getEdges().get(neighborId) > radius * (1 + eps)) continue;
avg. precision@100: 0.94325 at  246 queries/sec (avg  4.057ms/query), avg cals/search 657.955 approaching 1358.831 exploring, avg hops/search 27.878 approaching 90.805 exploring
avg. precision@100: 0.97427 at  247 queries/sec (avg  4.044ms/query), avg cals/search 662.392 approaching 2130.915 exploring, avg hops/search 27.948 approaching 143.984 exploring
avg. precision@100: 0.99246 at  139 queries/sec (avg  7.154ms/query), avg cals/search 666.193 approaching 3549.076 exploring, avg hops/search 27.991 approaching 248.087 exploring
avg. precision@100: 0.99567 at  144 queries/sec (avg  6.898ms/query), avg cals/search 667.427 approaching 4283.990 exploring, avg hops/search 28.007 approaching 305.190 exploring

	 * if(nextVertex.getVertex().getEdges().get(neighborId) > radius) continue;
avg. precision@100: 0.94278 at  265 queries/sec (avg  3.769ms/query), avg cals/search 656.806 approaching 1350.524 exploring, avg hops/search 27.867 approaching 90.812 exploring
avg. precision@100: 0.97249 at  253 queries/sec (avg  3.941ms/query), avg cals/search 656.806 approaching 2072.177 exploring, avg hops/search 27.867 approaching 143.940 exploring
avg. precision@100: 0.99040 at  159 queries/sec (avg  6.270ms/query), avg cals/search 656.806 approaching 3374.394 exploring, avg hops/search 27.867 approaching 247.607 exploring
avg. precision@100: 0.99372 at  147 queries/sec (avg  6.759ms/query), avg cals/search 656.806 approaching 4042.615 exploring, avg hops/search 27.867 approaching 304.379 exploring

	 * 
	 * 
	 * 
	 * Skip none-RNG during approach
avg. precision@100: 0.93984 at   81 queries/sec (avg 12.300ms/query), avg cals/search 569.448 approaching 1513.133 exploring, avg hops/search 34.239 approaching 85.519 exploring
avg. precision@100: 0.97379 at   64 queries/sec (avg 15.501ms/query), avg cals/search 569.448 approaching 2355.194 exploring, avg hops/search 34.239 approaching 138.531 exploring
avg. precision@100: 0.99277 at   40 queries/sec (avg 24.534ms/query), avg cals/search 569.448 approaching 3879.340 exploring, avg hops/search 34.239 approaching 242.711 exploring
avg. precision@100: 0.99594 at   35 queries/sec (avg 28.161ms/query), avg cals/search 569.448 approaching 4661.463 exploring, avg hops/search 34.239 approaching 299.891 exploring
	
avg. precision@100: 0.92831 at   85 queries/sec (avg 11.656ms/query), avg cals/search 526.354 approaching 1483.708 exploring, avg hops/search 38.316 approaching 83.172 exploring
avg. precision@100: 0.96933 at   71 queries/sec (avg 13.998ms/query), avg cals/search 526.354 approaching 2324.850 exploring, avg hops/search 38.316 approaching 135.424 exploring
avg. precision@100: 0.99176 at   45 queries/sec (avg 22.220ms/query), avg cals/search 526.354 approaching 3858.653 exploring, avg hops/search 38.316 approaching 239.201 exploring
avg. precision@100: 0.99548 at   37 queries/sec (avg 26.585ms/query), avg cals/search 526.354 approaching 4645.948 exploring, avg hops/search 38.316 approaching 296.327 exploring

avg. precision@100: 0.92944 at   82 queries/sec (avg 12.174ms/query), avg cals/search 476.785 approaching 1544.764 exploring
avg. precision@100: 0.96960 at   72 queries/sec (avg 13.729ms/query), avg cals/search 476.785 approaching 2389.737 exploring
avg. precision@100: 0.99175 at   44 queries/sec (avg 22.279ms/query), avg cals/search 476.785 approaching 3925.014 exploring
avg. precision@100: 0.99543 at   38 queries/sec (avg 25.991ms/query), avg cals/search 476.785 approaching 4712.033 exploring
	 * 
	 * Skip none-RNG during approach and RNG during explore
avg. precision@100: 0.78952 at   88 queries/sec (avg 11.238ms/query), avg cals/search 569.448 approaching 639.152 exploring
avg. precision@100: 0.83444 at   68 queries/sec (avg 14.500ms/query), avg cals/search 569.448 approaching 949.388 exploring
avg. precision@100: 0.88483 at   48 queries/sec (avg 20.525ms/query), avg cals/search 569.448 approaching 1508.616 exploring
avg. precision@100: 0.90194 at   40 queries/sec (avg 24.951ms/query), avg cals/search 569.448 approaching 1801.564 exploring

avg. precision@100: 0.81266 at   87 queries/sec (avg 11.415ms/query), avg cals/search 526.354 approaching 805.072 exploring
avg. precision@100: 0.86881 at   75 queries/sec (avg 13.223ms/query), avg cals/search 526.354 approaching 1215.723 exploring
avg. precision@100: 0.92235 at   48 queries/sec (avg 20.656ms/query), avg cals/search 526.354 approaching 1966.619 exploring
avg. precision@100: 0.93849 at   40 queries/sec (avg 24.533ms/query), avg cals/search 526.354 approaching 2358.398 exploring

avg. precision@100: 0.80793 at   91 queries/sec (avg 10.935ms/query), avg cals/search 476.785 approaching 867.041 exploring
avg. precision@100: 0.86384 at   72 queries/sec (avg 13.736ms/query), avg cals/search 476.785 approaching 1283.359 exploring
avg. precision@100: 0.91890 at   49 queries/sec (avg 20.315ms/query), avg cals/search 476.785 approaching 2038.175 exploring
avg. precision@100: 0.93551 at   41 queries/sec (avg 24.081ms/query), avg cals/search 476.785 approaching 2431.010 exploring


*
*
	 * @param deg
	 * @param siftBaseDir
	 * @throws IOException
	 */
	public static void testGraph(DynamicExplorationGraph deg, Path siftBaseDir) throws IOException {

		// load query data
		float[][] queryData = Sift1M.loadQueryData(siftBaseDir);
		int[][] groundtruthData = Sift1M.loadGroundtruthData(siftBaseDir);
		
		// test graph
		final int k = 100;	
//		for(float eps : new float[]{0.125f, 0.13f, 0.135f, 0.14f}) {	
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
			float aDistCals = (float) ArrayBasedWeightedUndirectedRegularGraph.ApproachDistCalcs.floatValue() / queryData.length;
			float edistCals = ((float) ArrayBasedWeightedUndirectedRegularGraph.ExploreDistCalcs.floatValue() / queryData.length);
			float aHopCount = (float) ArrayBasedWeightedUndirectedRegularGraph.ApproachHopCount.floatValue() / queryData.length;
			float eHopCount = ((float) ArrayBasedWeightedUndirectedRegularGraph.ExploreHopCount.floatValue() / queryData.length);
			float meanElapsedTime = elapsedMilliseconds.floatValue() / queryData.length;
			float meanPrecisionAtK = precisionAtK.floatValue() / k / queryData.length;
			int queriesPerSecond = (int)(1000 / meanElapsedTime);
			System.out.printf("avg. precision@%3d: %.5f at %4d queries/sec (avg %6.3fms/query), avg cals/search %6.3f approaching %6.3f exploring, avg hops/search %6.3f approaching %6.3f exploring\n", k, meanPrecisionAtK, queriesPerSecond, meanElapsedTime, aDistCals, edistCals, aHopCount, eHopCount);
			
			ArrayBasedWeightedUndirectedRegularGraph.ExploreDistCalcs.set(0);
			ArrayBasedWeightedUndirectedRegularGraph.ApproachDistCalcs.set(0);
			ArrayBasedWeightedUndirectedRegularGraph.ExploreHopCount.set(0);
			ArrayBasedWeightedUndirectedRegularGraph.ApproachHopCount.set(0);
		}
	}
}
