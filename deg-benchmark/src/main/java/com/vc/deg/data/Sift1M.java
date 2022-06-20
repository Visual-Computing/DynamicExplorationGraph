package com.vc.deg.data;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;


public class Sift1M {

	public static final float[][] loadQueryData(Path baseDir) throws IOException {
		return fvecs_read(baseDir.resolve("sift_query.fvecs"));
	}
	
	public static final float[][] loadLearnData(Path baseDir) throws IOException {
		return fvecs_read(baseDir.resolve("sift_learn.fvecs"));
	}
	
	public static final float[][] loadBaseData(Path baseDir) throws IOException {
		return fvecs_read(baseDir.resolve("sift_base.fvecs"));
	}
	
	public static final int[][] loadGroundtruthData(Path baseDir) throws IOException {
		return ivecs_read(baseDir.resolve("sift_groundtruth.ivecs"));
	}
	
	/**
	 * http://corpus-texmex.irisa.fr/fvecs_read.m
	 * 
	 * @param feature_file
	 * @return
	 * @throws IOException 
	 */
	public static float[][] fvecs_read(Path feature_file) throws IOException {    
	    List<float[]> features = new ArrayList<>();
		try(SeekableByteChannel ch = Files.newByteChannel(feature_file, StandardOpenOption.READ)) {
			while(ch.position() < ch.size()) {
				
				// read the size of the feature vector
				ByteBuffer countBuffer = ByteBuffer.allocate(4);
				countBuffer.order(ByteOrder.LITTLE_ENDIAN);
				ch.read(countBuffer);
				countBuffer.flip();
			    int c = countBuffer.getInt();
			    
			    // read the values of the feature
			    float[] feature = new float[c];
				ByteBuffer valueBuffer = ByteBuffer.allocate(4 * c);
				ch.read(valueBuffer);
				valueBuffer.flip();
				valueBuffer.order(ByteOrder.LITTLE_ENDIAN);
				valueBuffer.asFloatBuffer().get(feature);
			    features.add(feature);
			}
		}
		
		return features.toArray(new float[features.size()][]);
	}
	
	/**
	 * http://corpus-texmex.irisa.fr/ivecs_read.m
	 * 
	 * @param feature_file
	 * @return
	 * @throws IOException 
	 */
	public static int[][] ivecs_read(Path feature_file) throws IOException {    
	    List<int[]> features = new ArrayList<>();
		try(SeekableByteChannel ch = Files.newByteChannel(feature_file, StandardOpenOption.READ)) {
			while(ch.position() < ch.size()) {
				
				// read the size of the feature vector
				ByteBuffer countBuffer = ByteBuffer.allocate(4);
				countBuffer.order(ByteOrder.LITTLE_ENDIAN);
				ch.read(countBuffer);
				countBuffer.flip();
			    int c = countBuffer.getInt();
			    
			    // read the values of the feature
			    int[] feature = new int[c];
				ByteBuffer valueBuffer = ByteBuffer.allocate(4 * c);
				ch.read(valueBuffer);
				valueBuffer.flip();
				valueBuffer.order(ByteOrder.LITTLE_ENDIAN);
				valueBuffer.asIntBuffer().get(feature);
			    features.add(feature);
			}
		}
		
		return features.toArray(new int[features.size()][]);
	}
	
}
