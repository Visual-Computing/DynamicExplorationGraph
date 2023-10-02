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
import com.vc.deg.GraphFactory;
import com.vc.deg.data.GloVe;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.FloatL2Space;
import com.vc.deg.graph.GraphDesigner;

/**
 * 
 * 
 * @author Nico Hezel
 */
public class GraphReductionBenchmark {

	public static void main(String[] args) throws IOException, ClassNotFoundException {

		
		// register the feature space needed in the graph
		final AtomicInteger distanceCalculationCount = new AtomicInteger();
		FeatureSpace space = new FloatL2Space(100) { 
			@Override
			public float computeDistance(FeatureVector f1, FeatureVector f2) {
				distanceCalculationCount.incrementAndGet();
				return super.computeDistance(f1, f2);
			}
		};		
		FeatureSpace.registerFeatureSpace(space);

		// load the graph
		final Path graphFile = Paths.get("e:\\Data\\Feature\\GloVe\\deg\\online\\100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg");
		System.out.println("Load the graph "+graphFile);
		final DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile, "float");
		System.out.println("Use DEG"+deg.edgePerVertex()+" with "+deg.size()+" vertices");
				
		// remove the second half of the base data from the graph
		final GraphDesigner designer = deg.designer();
		final int initialSize = deg.size();
		final int finalSize = initialSize/2;
		for (int i = 0; i < finalSize; i++) 
			designer.remove(finalSize + i);

		// change designer settings
		final Random rnd = new Random();
		final int edgesPerNode = deg.edgePerVertex();
		designer.setRandom(rnd);
		designer.setExtendK(edgesPerNode);
		designer.setExtendEps(0.2f);
		designer.setImproveK(edgesPerNode);
		designer.setImproveEps(0.001f);
		designer.setMaxPathLength(5);
		
		// start the build process
		final AtomicLong start = new AtomicLong(System.currentTimeMillis());
		final AtomicLong durationMs = new AtomicLong(0);
		designer.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
			final int size = (int)(initialSize+added)-(int)removed;
			if(step % 10000 == 0) {
				durationMs.addAndGet(System.currentTimeMillis() - start.get());
				final float avgEdgeWeight = designer.calcAvgEdgeWeight() * 1000; // scaler
				final boolean valid = designer.checkGraphValidation(size, edgesPerNode);
				final int duration = (int)(durationMs.get() / 1000);
				System.out.printf("Step %7d, %3ds, Q: %4.2f, Size %7d, Added %7d (last id %3d), Removed %7d (last id %3d), improved %3d, tries %3d, valid %s\n", step, duration, avgEdgeWeight, size, added, lastAdd, removed, lastRemoved, improved, tries, Boolean.toString(valid));
				start.set(System.currentTimeMillis());
			}
			
			// stop the building process
			if(size == finalSize)
				designer.stop();
		});
		
		System.out.println("Finished");
	}	
}