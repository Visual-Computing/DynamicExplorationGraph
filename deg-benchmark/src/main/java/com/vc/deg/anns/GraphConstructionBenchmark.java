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
 * 
 * 
 * @author Nico Hezel
 */
public class GraphConstructionBenchmark {

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
		
	}
}
