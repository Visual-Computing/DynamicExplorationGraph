package com.vc.deg.anns;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.GraphFactory;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.FloatL2Space;
import com.vc.deg.graph.GraphDesigner;

/**
 * Compute the stats of the graphs
 * 
 * @author Nico Hezel
 */
public class GraphStatsBenchmark {

	protected final static Path siftBaseDir = Paths.get("C:/Data/Feature/SIFT1M/SIFT1M/");
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\java\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.float.deg");
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.rng_optimized.deg");

	// fast and good 1h
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd3+2_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.deg");
	
	// DEG30 (fast opt rng high) 			28min
	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.test.deg");

	// DEG30 (best opt rng) 			5h 31min
//	protected final static Path graphFile   = Paths.get("c:\\Data\\Feature\\SIFT1M\\deg\\best_distortion_decisions\\128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd15+15_improveTheBetterHalfOfTheNonPerfectEdges_RNGIncompleteAddMinimalSwap.deg");

	
	
	
//	protected final static Path graphFile   = Paths.get("C:/Data/Feature/2DGraph/L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg");
//	protected final static Path graphFile   = Paths.get("C:/Data/Feature/2DGraph/L2_rng.deg");
//	protected final static Path graphFile   = Paths.get("C:/Data/Feature/2DGraph/L2_dg.deg");
//	protected final static Path graphFile   = Paths.get("C:/Data/Feature/2DGraph/L2_K3_knng.deg");
//	protected final static Path graphFile   = Paths.get("C:/Data/Feature/2DGraph/L2_K3_knnAproxRNG.deg");

	
	public static void main(String[] args) throws IOException {
		
		int[][] top1000 = Sift1M.loadBaseTop1000(siftBaseDir);
		
		// register the feature space needed in the graph
		FeatureSpace.registerFeatureSpace(new FloatL2Space(128));
		FeatureSpace.registerFeatureSpace(new FloatL2Space(2));
		
		// load graph
		DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile, float.class.getSimpleName());
		GraphDesigner designer = deg.designer();
		System.out.println("AEW of "+graphFile.getFileName()+" is "+designer.calcAvgEdgeWeight());
		System.out.println("ANR of "+graphFile.getFileName()+" is "+designer.calcAvgNeighborRank(top1000));
		System.out.println("ANR of "+graphFile.getFileName()+" is "+designer.calcAvgNeighborRank());
	}
}
