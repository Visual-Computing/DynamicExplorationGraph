package com.vc.deg.anns;

import java.io.DataInput;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphFactory;
import com.vc.deg.data.Sift1M;
import com.vc.deg.feature.BinaryFeature;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.FloatL2Space;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.VertexCursor;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphFilterBenchmark {

	protected final static Path graphFile   = Paths.get("c:\\Data\\Images\\navigu\\pixabay\\graph\\visualGraph\\gpretfv\\layer0.binary.deg");

	public static void main(String[] args) throws IOException {
		
		final FeatureSpace GPRetSpace = new BinaryFeatureSpace(1024);
		final FeatureFactory GPRetFactory = new BinaryFeatureFactory(1024);
		
		FeatureSpace.registerFeatureSpace(GPRetSpace);
		FeatureVector.registerFeatureFactor(GPRetFactory);
		
		long start = System.currentTimeMillis();
		DynamicExplorationGraph deg = GraphFactory.getDefaultFactory().loadGraph(graphFile);
		System.out.println("loading graph took "+(System.currentTimeMillis()-start)+"ms");		
		
		for (int i = 0; i < 10; i++) {
			start = System.currentTimeMillis();
			
			final IntSet validIds = HashIntSets.newMutableSet(c -> {
				final VertexCursor cursor = deg.vertexCursor();
				while(cursor.moveNext())
					c.accept(cursor.getVertexLabel());			
			}, deg.size());
			final GraphFilter filter = new GraphFilter() {
				
				@Override
				public int size() {
					return validIds.size();
				}
				
				@Override
				public boolean isValid(int label) {
					return validIds.contains(label);
				}
			};
			System.out.println("building filter took "+(System.currentTimeMillis()-start)+"ms");
			start = System.currentTimeMillis();
			
			final int[] queryIds = new int[] { 1551664, 959394, 1675402};
			final int desiredCount = 200;
			final int[] exploreResult = deg.explore(queryIds, desiredCount, 0.0f, filter);
			System.out.println("exploring graph took "+(System.currentTimeMillis()-start)+"ms");
	
			
			System.out.println("Graph size: "+deg.size()+", exploration count: "+exploreResult.length);
		}
	}
	
	
	public static class BinaryFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		protected final int size;
		
		public BinaryFeatureFactory(int dims) {
			this.dims = dims;
			this.size = (int) Math.ceil((float)dims / 64);
		}
		
		@Override
		public Class<?> getComponentType() {
			return BinaryFeature.ComponentType;
		}

		@Override
		public int featureSize() {
			return size * Long.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final long[] feature = new long[size];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readLong();
			return new BinaryFeature(dims, feature);
		}
	}
	
	public static class BinaryFeatureSpace implements FeatureSpace {
		
		
		protected final int dims;
		protected final int size;
		
		public BinaryFeatureSpace(int dims) {
			this.dims = dims;
			this.size = (int) Math.ceil((float)dims / 64);
		}
		
		@Override
		public int featureSize() {
			return size * Long.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

		@Override
		public Class<?> getComponentType() {
			return BinaryFeature.ComponentType;
		}

		@Override
		public int metric() {
			return Metric.Manhatten.getId();
		}

		@Override
		public boolean isNative() {
			return false;
		}

		@Override
		public float computeDistance(FeatureVector f1, FeatureVector f2) {
			int sum = 0;
			for (int i = 0; i < size; i++) 
				sum += Long.bitCount(f1.readLong(i * Long.BYTES) ^ f2.readLong(i * Long.BYTES));
			return sum;
		}
	}
}
