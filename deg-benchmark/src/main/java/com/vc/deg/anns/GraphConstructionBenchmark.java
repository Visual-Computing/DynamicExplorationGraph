package com.vc.deg.anns;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphDesigner;
import com.vc.deg.GraphFactory;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.FloatL2Space;

/**
 * 
 * 
 * @author Nico Hezel
 */
public class GraphConstructionBenchmark {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		// load query data
		final Path siftBaseDir = Paths.get("C:/Data/Feature/SIFT1M/SIFT1M/");
		System.out.println("Load the dataset "+siftBaseDir);
		final float[][] baseData = Sift1M.loadBaseData(siftBaseDir);
		
		// register the feature space needed in the graph
		final AtomicInteger distanceCalculationCount = new AtomicInteger();
		FeatureSpace space = new FloatL2Space(128) { 
			@Override
			public float computeDistance(FeatureVector f1, FeatureVector f2) {
				distanceCalculationCount.incrementAndGet();
				return super.computeDistance(f1, f2);
			}
		};
		
		// hierarchical 
		final int edgesPerNode = 30;
		final int topRankSize = 400;
		final Path graphFile = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_RNGMinimalAdd\\");
		final DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().newHierchicalGraph(space, edgesPerNode, topRankSize);
		
		// simple DEG
//		final int edgesPerNode = 30;
//		final Path graphFile = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_RNGMinimalAdd.java.float.deg");
//		final DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().newGraph(space, edgesPerNode);
		
		// give all base data to the designer to be added to the graph during the build process 
		int maxIndex = 0;
		final Random rnd = new Random(7);
		final GraphDesigner designer = deg.designer();
		
		// random add
		if(maxIndex > 0) {
			final List<Integer> availableIdx = IntStream.range(0, maxIndex).mapToObj(Integer::valueOf).collect(Collectors.toList());
			final List<Integer> unavailableIdx = new ArrayList<>();
			for (int it = 0; it < 100; it++) {
				
				Collections.shuffle(availableIdx, rnd);
				final int add = rnd.nextInt(availableIdx.size());
				for (int i = 0; i < add; i++) {
					final int index = availableIdx.remove(availableIdx.size()-1);
					designer.add(index, new FloatFeature(baseData[index]));
					unavailableIdx.add(index);
				}
					
				Collections.shuffle(unavailableIdx, rnd);
				final int remove = rnd.nextInt(unavailableIdx.size());
				for (int i = 0; i < remove; i++)  {
					final int index = unavailableIdx.remove(unavailableIdx.size()-1);
					designer.remove(index);
					availableIdx.add(index);
				}
			}		
			for(int i : availableIdx)
				designer.add(i, new FloatFeature(baseData[i]));
		}
		
		// add all available data points to the graph
		for (int i = maxIndex; i < baseData.length; i++)
			designer.add(i, new FloatFeature(baseData[i]));
		
		
		// change designer settings
		designer.setRandom(rnd);
		designer.setExtendK(60);
		designer.setExtendEps(0.2f);
		designer.setImproveK(30);
		designer.setImproveEps(0.001f);
		designer.setMaxPathLength(5);
		
		// start the build process
		final AtomicLong start = new AtomicLong(System.currentTimeMillis());
		final AtomicLong durationMs = new AtomicLong(0);
		designer.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
			final int size = (int)added-(int)removed;
			if(step % 100 == 0) {
				durationMs.addAndGet(System.currentTimeMillis() - start.get());
				final float avgEdgeWeight = designer.calcAvgEdgeWeight();
				final boolean valid = designer.checkGraphValidation(size, edgesPerNode);
				final int duration = (int)(durationMs.get() / 1000);
				System.out.printf("Step %7d, %3ds, Q: %4.2f, Size %7d, Added %7d (last id %3d), Removed %7d (last id %3d), improved %3d, tries %3d, valid %s\n", step, duration, avgEdgeWeight, size, added, lastAdd, removed, lastRemoved, improved, tries, Boolean.toString(valid));
				start.set(System.currentTimeMillis());
			}
			
			if(step % 1000 == 0) {
				try {
					deg.writeToFile(graphFile);
				} catch (ClassNotFoundException | IOException e) {
					e.printStackTrace();
				}
			}
			
			// stop the building process
			if(added == baseData.length)
				designer.stop();
		});
		
		// store and test the graph
		deg.writeToFile(graphFile);
		GraphSearchBenchmark.testGraph(deg, siftBaseDir);
		
		System.out.println("Finished");
	}	
}