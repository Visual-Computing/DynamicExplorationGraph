package com.vc.deg.data;

import java.io.IOException;
import java.nio.file.Path;


public class Graph2DData {

	
	public static final float[][] loadBaseData(Path baseDir) throws IOException {
		return Sift1M.fvecs_read(baseDir.resolve("base.fvecs"));
	}

	public static final float[][] loadQueryData(Path baseDir) throws IOException {
		return Sift1M.fvecs_read(baseDir.resolve("query.fvecs"));
	}
	
	public static final int[][] loadGroundtruthData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("query_gt.ivecs"));
	}
	
	public static final float[][] loadExploreData(Path baseDir) throws IOException {
		return Sift1M.fvecs_read(baseDir.resolve("explore.fvecs"));
	}
	
	public static final int[][] loadExploreQueryData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("explore_entry_node.ivecs"));
	}
	
	public static final int[][] loadExploreGroundtruthData(Path baseDir) throws IOException {
		return Sift1M.ivecs_read(baseDir.resolve("explore_gt.ivecs"));
	}
}
