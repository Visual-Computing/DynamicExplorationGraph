package com.vc.deg.data;

import java.io.IOException;
import java.nio.file.Path;


public class GloVe {

	
	public static final float[][] loadBaseData(Path baseDir) throws IOException {
		return Sift1M.fvecs_read(baseDir.resolve("glove-100_base.fvecs"));
	}
	
	public static final float[][] loadQueryData(Path baseDir) throws IOException {
		return Sift1M.fvecs_read(baseDir.resolve("glove-100_query.fvecs"));
	}
	
	public static final int[][] loadGroundtruthData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("glove-100_groundtruth.ivecs"));
	}
	
	public static final int[][] loadGroundtruthDataBase591757(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("glove-100_groundtruth_base591757.ivecs"));
	}
	
	public static final int[][] loadExploreQueryData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("glove-100_explore_entry_vertex.ivecs"));
	}
	
	public static final int[][] loadExploreGroundtruthData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("sglove-100_explore_ground_truth.ivecs"));
	}
}
