package com.vc.deg.anns;

import java.io.DataInput;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;
import java.util.function.IntConsumer;
import java.util.function.Supplier;

import org.roaringbitmap.RoaringBitmap;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphFactory;
import com.vc.deg.HierarchicalDynamicExplorationGraph;
import com.vc.deg.feature.BinaryFeature;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.VertexCursor;

/**
 * Benchmark of the graph against the SIFT1M dataset
 * 
 * @author Nico Hezel
 */
public class GraphFilterBenchmark {

	protected final static Path graphFile = Paths.get("c:\\Data\\Images\\navigu\\pixabay\\graph\\visualGraph\\gpretfv\\");

	public static void main(String[] args) throws IOException {

		final FeatureSpace GPRetSpace = new BinaryFeatureSpace(1024);
		final FeatureFactory GPRetFactory = new BinaryFeatureFactory(1024);

		FeatureSpace.registerFeatureSpace(GPRetSpace);
		FeatureVector.registerFeatureFactor(GPRetFactory);

		long start = System.currentTimeMillis();
		final HierarchicalDynamicExplorationGraph hdeg = GraphFactory.getDefaultFactory().loadHierchicalGraph(graphFile);
		final DynamicExplorationGraph deg = hdeg.getGraph(0);
		System.out.println("loading graph took "+(System.currentTimeMillis()-start)+"ms");
		
		// represents elements on the global map
		final int globalMapElementCount = 20000;
		final Random rnd = new Random(7);
		final IntSet globalMapIds = HashIntSets.newMutableSet();
		while(globalMapIds.size() < globalMapElementCount) 
			globalMapIds.add(deg.getRandomLabel(rnd));
		final int[] globalMapIdArray = globalMapIds.toIntArray();

		// prepare query data in order to compare the results
		final int desiredCount = 1000;
		final float eps = 0.0f;
		final int queryCount = 3;
		final IntSet queryIdsSet = HashIntSets.newMutableSet();
		while(queryIdsSet.size() < queryCount) 
			queryIdsSet.add(hdeg.getGraph(hdeg.levelCount()-1).getRandomLabel(rnd));
		final int[] queryIds = queryIdsSet.toIntArray(); // new int[] { 1551664, 959394, 1675402 };

		// how many time should the test be repeated?
		final int testCount = 50;
		

//		filterChainHashSetTest(testCount, hdeg, gloabelMapIdArray, queryIds, eps, desiredCount);
//		System.out.println("\n\n---------------------------------------------------------------------------------------------------------\n\n");
//		roaringBitmapTest(testCount, hdeg, gloabelMapIdArray, queryIds, eps, desiredCount);
//		System.out.println();
//		noGlobalFilterBitmapTest(testCount, deg, gloabelMapIdArray, queryIds, eps, desiredCount);
//		System.out.println("\n\n---------------------------------------------------------------------------------------------------------\n\n");
//		bitsetTest(testCount, deg, gloabelMapIdArray, queryIds, eps, desiredCount);
//		System.out.println("\n\n---------------------------------------------------------------------------------------------------------\n\n");
		hashSetTest(testCount, deg, globalMapIdArray, queryIds, eps, desiredCount);
		System.out.println();
		noGlobalFilterHashSetTest(testCount, deg, globalMapIdArray, queryIds, eps, desiredCount);
		System.out.println();
		noGlobalFilterExcludingHashSetTest(testCount, deg, globalMapIdArray, queryIds, eps, desiredCount);
	}

	// ---------------------------------------------------------------------------------------------------
	// ------------------------------------- Chain HashSets ----------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	public static void filterChainHashSetTest(int testCount, HierarchicalDynamicExplorationGraph hdeg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {

		final DynamicExplorationGraph deg = hdeg.getGraph(0);
		final int[] allIdsShuffled = shuffle(getAllIds(deg), 5);
		for (int l = 0; l < hdeg.levelCount(); l++) {
			final DynamicExplorationGraph ldeg = hdeg.getGraph(l);
			System.out.println("filterChainHashSetTest Level "+l+" (Graph size: "+ldeg.size()+"), globalMap exludes "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

			for (int p : new int[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100}) {
				final IntSet filteredIds = HashIntSets.newMutableSet(Arrays.copyOf(allIdsShuffled, (int)((float)(p)/100f*deg.size())));
				final GraphFilter globalFilter = new GraphFilter() {

					@Override
					public int size() {
						return filteredIds.size();
					}

					@Override
					public float getInclusionRate() {
						return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
					}

					@Override
					public boolean isValid(int label) {
						return filteredIds.contains(label);
					}

					@Override
					public void forEachValidId(IntConsumer action) {
						filteredIds.forEach(action);
					}
				};

				final Supplier<GraphFilter> filterBuilder = () -> {

					// the global filter might contains ids which are not in the given graph level
					final IntSet invalidIds = HashIntSets.newImmutableSet(excludeIds);
					final GraphFilter filterChain = new GraphFilter() {

						@Override
						public int size() {
							return globalFilter.size();
						}

						@Override
						public float getInclusionRate() {
							return globalFilter.getInclusionRate();
						}

						@Override
						public boolean isValid(int label) {
							return globalFilter.isValid(label) && invalidIds.contains(label) == false;
						}

						@Override
						public void forEachValidId(IntConsumer action) {
							globalFilter.forEachValidId(id -> {
								if(invalidIds.contains(id) == false)
									action.accept(id);
							});
						}
					};
					return filterChain;
				};

				final IntSet validIds = HashIntSets.newMutableSet(filteredIds);
				for (int excludeId : excludeIds) 
					validIds.removeInt(excludeId);
				System.out.printf("%3d%% include %6d ", p, validIds.size());
				filterTest(filterBuilder, globalFilter, testCount, ldeg, excludeIds, queryIds, eps, desiredCount);
			}
			System.out.println();
		}
	}
	

	// ---------------------------------------------------------------------------------------------------
	// ------------------------------------- Roaring Bitmap ----------------------------------------------
	// ---------------------------------------------------------------------------------------------------
	
	public static void roaringBitmapTest(int testCount, HierarchicalDynamicExplorationGraph hdeg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {

		final DynamicExplorationGraph deg = hdeg.getGraph(0);
		final int[] allIdsShuffled = shuffle(getAllIds(deg), 5);
		for (int l = 0; l < hdeg.levelCount(); l++) {
			final DynamicExplorationGraph ldeg = hdeg.getGraph(l);
			System.out.println("roaringBitmapTest Level "+l+" (Graph size: "+ldeg.size()+"), globalMap exludes "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

			for (int p : new int[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100}) {
				final int[] subsetIds = Arrays.copyOf(allIdsShuffled, (int)((float)(p)/100f*deg.size()));
				final RoaringBitmap filteredIds = RoaringBitmap.bitmapOfUnordered(subsetIds);
				final int filteredIdsSize = filteredIds.getCardinality();
				final GraphFilter globalFilter = new GraphFilter() {

					@Override
					public int size() {
						return filteredIdsSize;
					}

					@Override
					public float getInclusionRate() {
						return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
					}

					@Override
					public boolean isValid(int label) {
						return filteredIds.contains(label);
					}

					@Override
					public void forEachValidId(IntConsumer action) {
						filteredIds.forEach((int i) -> action.accept(i));
					}
				};
				
				final Supplier<GraphFilter> filterBuilder = () -> {

					// the global filter might contains ids which are not in the given graph level
					final RoaringBitmap validIds = filteredIds.clone(); 
					for (int excludeId : excludeIds) 
						validIds.remove(excludeId);
					final int validIdsSize = validIds.getCardinality();
					final GraphFilter filter = new GraphFilter() {

						@Override
						public int size() {
							return validIdsSize;
						}

						@Override
						public float getInclusionRate() {
							return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
						}

						@Override
						public boolean isValid(int label) {
							return validIds.contains(label);
						}

						@Override
						public void forEachValidId(IntConsumer action) {
							validIds.forEach((int i) -> action.accept(i));
						}
					};
					return filter;
				};

				final RoaringBitmap validIds = filteredIds.clone(); 
				for (int excludeId : excludeIds) 
					validIds.remove(excludeId);
				final int validIdsSize = validIds.getCardinality();
				System.out.printf("%3d%% include %6d ", p, validIdsSize);
				filterTest(filterBuilder, globalFilter, testCount, ldeg, excludeIds, queryIds, eps, desiredCount);
			}
		}
	}
	
	public static void noGlobalFilterBitmapTest(int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {
		System.out.println("noGlobalFilterBitmapTest (Graph size: "+deg.size()+") exlude "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

		final RoaringBitmap degLabelsBitmap = new RoaringBitmap();
		{
			final VertexCursor cursor = deg.vertexCursor();
			while(cursor.moveNext()) {
				final int label = cursor.getVertexLabel();
				degLabelsBitmap.add(label);
			}
		}

		final Supplier<GraphFilter> filterBuilder = () -> {
			final RoaringBitmap validIds = degLabelsBitmap.clone();
			for (int excludeId : excludeIds) 
				validIds.remove(excludeId);
			final int validIdsSize = validIds.getCardinality();
			final GraphFilter filter = new GraphFilter() {

				@Override
				public int size() {
					return validIdsSize;
				}

				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}

				@Override
				public boolean isValid(int label) {
					return validIds.contains(label);
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					validIds.forEach((int i) -> action.accept(i));
				}
			};
			return filter;
		};
		filterTest(filterBuilder, null, testCount, deg, excludeIds, queryIds, eps, desiredCount);
	}
	

	// ---------------------------------------------------------------------------------------------------
	// --------------------------------------------- BitSet ----------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	public static void bitsetTest(int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {
		System.out.println("bitsetTest (Graph size: "+deg.size()+"), globalMap exludes "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

		final int[] allIdsShuffled = shuffle(getAllIds(deg), 5);
		int max = 0;
		for (int id : allIdsShuffled)
			max = Math.max(max, id);

		for (int p : new int[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100}) {
			final int[] subsetIds = Arrays.copyOf(allIdsShuffled, (int)((float)(p)/100f*deg.size()));
			final BitSet filteredIds = new BitSet(max);
			for (int id : subsetIds)
				filteredIds.set(id);
			final int filteredIdsSize = filteredIds.cardinality();
			final GraphFilter globalFilter = new GraphFilter() {

				@Override
				public int size() {
					return filteredIdsSize;
				}

				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}

				@Override
				public boolean isValid(int label) {
					return filteredIds.get(label);
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					filteredIds.stream().forEach(action);
				}
			};

			final Supplier<GraphFilter> filterBuilder = () -> {

				// the global filter might contains ids which are not in the given graph level
				final BitSet validIds = (BitSet) filteredIds.clone();
				for (int excludeId : excludeIds) 
					validIds.clear(excludeId);
				final int validIdsSize = validIds.cardinality();
				final GraphFilter filter = new GraphFilter() {

					@Override
					public int size() {
						return validIdsSize;
					}

					@Override
					public float getInclusionRate() {
						return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
					}

					@Override
					public boolean isValid(int label) {
						return validIds.get(label);
					}

					@Override
					public void forEachValidId(IntConsumer action) {
						validIds.stream().forEach(action);
					}
				};
				return filter;
			};

			final BitSet validIds = (BitSet) filteredIds.clone();
			for (int excludeId : excludeIds) 
				validIds.clear(excludeId);
			final int validIdsSize = validIds.cardinality();
			System.out.printf("%3d%% include %6d ", p, validIdsSize);
			filterTest(filterBuilder, globalFilter, testCount, deg, excludeIds, queryIds, eps, desiredCount);
		}
	}
	
	// ---------------------------------------------------------------------------------------------------
	// --------------------------------------------- HashSet ---------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	public static void hashSetTest(int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {

		System.out.println("hashSetTest (Graph size: "+deg.size()+"), globalMap exludes "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");
		final int[] allIdsShuffled = shuffle(getAllIds(deg), 5);
		for (int p : new int[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100}) {
			final IntSet filteredIds = HashIntSets.newMutableSet(Arrays.copyOf(allIdsShuffled, (int)((float)(p)/100f*deg.size())));
			final GraphFilter globalFilter = new GraphFilter() {

				@Override
				public int size() {
					return filteredIds.size();
				}

				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}

				@Override
				public boolean isValid(int label) {
					return filteredIds.contains(label);
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					filteredIds.forEach(action);
				}
			};

			final Supplier<GraphFilter> filterBuilder = () -> {

				// the global filter might contains ids which are not in the given graph level
				final IntSet validIds = HashIntSets.newMutableSet(c -> {
					final VertexCursor cursor = deg.vertexCursor();
					while(cursor.moveNext()) {
						final int label = cursor.getVertexLabel();
						if(globalFilter.isValid(label))
							c.accept(label);
					}
				}, globalFilter.size());
				for (int excludeId : excludeIds) 
					validIds.removeInt(excludeId);
				final GraphFilter filter = new GraphFilter() {

					@Override
					public int size() {
						return validIds.size();
					}

					@Override
					public float getInclusionRate() {
						return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
					}

					@Override
					public boolean isValid(int label) {
						return validIds.contains(label);
					}

					@Override
					public void forEachValidId(IntConsumer action) {
						validIds.forEach(action);
					}
				};
				return filter;
			};

			final IntSet validIds = HashIntSets.newMutableSet(c -> {
				final VertexCursor cursor = deg.vertexCursor();
				while(cursor.moveNext()) {
					final int label = cursor.getVertexLabel();
					if(globalFilter.isValid(label))
						c.accept(label);
				}
			}, globalFilter.size());
			for (int excludeId : excludeIds) 
				validIds.removeInt(excludeId);
			System.out.printf("%3d%% include %6d ", p, validIds.size());
			filterTest(filterBuilder, globalFilter, testCount, deg, excludeIds, queryIds, eps, desiredCount);
		}
	}
	
	public static void noGlobalFilterHashSetTest(int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {
		System.out.println("noGlobalFilterHashSetTest (Graph size: "+deg.size()+") exlude "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

		final Supplier<GraphFilter> filterBuilder = () -> {
			final IntSet validIds = HashIntSets.newMutableSet(c -> {
				final VertexCursor cursor = deg.vertexCursor();
				while(cursor.moveNext())
					c.accept(cursor.getVertexLabel());			
			}, deg.size());
			for (int excludeId : excludeIds) 
				validIds.removeInt(excludeId);
			final GraphFilter filter = new GraphFilter() {

				@Override
				public int size() {
					return validIds.size();
				}

				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}

				@Override
				public boolean isValid(int label) {
					return validIds.contains(label);
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					validIds.forEach(action);
				}
			};
			return filter;
		};
		filterTest(filterBuilder, null, testCount, deg, excludeIds, queryIds, eps, desiredCount);
	}

	public static void noGlobalFilterExcludingHashSetTest(int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {
		System.out.println("noGlobalFilterExcludingHashSetTest (Graph size: "+deg.size()+") exlude "+excludeIds.length+" ids, with "+queryIds.length+" queries, eps "+eps+" and Top"+desiredCount+" results");

		final Supplier<GraphFilter> filterBuilder = () -> {
			final IntSet invalidIds = HashIntSets.newMutableSet(excludeIds);
			final GraphFilter filter = new GraphFilter() {

				@Override
				public int size() {
					return deg.size() - invalidIds.size();
				}

				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}

				@Override
				public boolean isValid(int label) {
					return !invalidIds.contains(label);
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					final VertexCursor cursor = deg.vertexCursor();
					while(cursor.moveNext()) {
						final int label = cursor.getVertexLabel();
						if(invalidIds.contains(label) == false)
							action.accept(label);
					}
				}
			};
			return filter;
		};

		filterTest(filterBuilder, null, testCount, deg, excludeIds, queryIds, eps, desiredCount);
	}
	

	// ---------------------------------------------------------------------------------------------------
	// --------------------------------------------- Helper ----------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	public static int[] shuffle(int[] values, int repeat) {
		final Random rand = new Random(7);

		for (int r = 0; r < repeat; r++) {			
			for (int i = 0; i < values.length; i++) {
				int randomIndexToSwap = rand.nextInt(values.length);
				int temp = values[randomIndexToSwap];
				values[randomIndexToSwap] = values[i];
				values[i] = temp;
			}
		}
		return values;
	}

	public static int[] getAllIds(DynamicExplorationGraph deg) {
		final int[] ids = new int[deg.size()];
		int pos = 0;
		final VertexCursor cursor = deg.vertexCursor();
		while(cursor.moveNext())
			ids[pos++] = cursor.getVertexLabel();
		return ids;
	}

	public static void filterTest(Supplier<GraphFilter> filterBuilder, GraphFilter globalFilter, int testCount, DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount) throws IOException {
		long sumFilterBuildTime = 0;
		long sumExploreTime = 0;
		long sumIsRight = 0;
		final int[] gt = computeGroundTruth(deg, excludeIds, queryIds, eps, desiredCount, globalFilter);
		for (int i = 0; i < testCount*2; i++) {

			long start = System.currentTimeMillis();
			final GraphFilter filter = filterBuilder.get();
			if(i > testCount)
				sumFilterBuildTime += System.currentTimeMillis()-start;

			start = System.currentTimeMillis();
			final int[] exploreResult = deg.explore(queryIds, desiredCount, eps, filter);
			if(i > testCount)
				sumExploreTime += System.currentTimeMillis()-start;

			final boolean isRight = checkResults(exploreResult, gt);
			if(i >= testCount && isRight)
				sumIsRight++;
		}
		System.out.printf("filter build time %6dms, explore time %6dms and the result is right %b\n",sumFilterBuildTime/testCount, sumExploreTime/testCount, sumIsRight/testCount < 1 ? false : true);
	}


	/**
	 * Check if the explore results are correct
	 * 
	 * @param exploreResult
	 * @param gt
	 * @return
	 * @throws IOException
	 */
	public static boolean checkResults(int[] exploreResult, int[] gt) throws IOException {

		if(gt.length != exploreResult.length)
			return false;
		for (int i = 0; i < gt.length; i++) 
			if(gt[i] != exploreResult[i])
				return false;
		return true;
	}

	/**
	 * Compute the ground truth results
	 * 
	 * @param deg
	 * @param excludeIds
	 * @param queryIds
	 * @param eps
	 * @param desiredCount
	 * @return
	 */
	public static int[] computeGroundTruth(DynamicExplorationGraph deg, int[] excludeIds, int[] queryIds, float eps, int desiredCount, GraphFilter globalFilter) {
		final IntSet invalidIds = HashIntSets.newMutableSet(excludeIds);
		final GraphFilter filter;
		if(globalFilter == null) {
			filter = new GraphFilter() {
	
				@Override
				public int size() {
					return deg.size() - invalidIds.size();
				}
	
				@Override
				public float getInclusionRate() {
					return Math.max(0, Math.min(1, ((float)size()) / deg.size()));
				}
	
				@Override
				public boolean isValid(int label) {
					return !invalidIds.contains(label);
				}
	
				@Override
				public void forEachValidId(IntConsumer action) {
				}
			};
		} else {
			filter = new GraphFilter() {

				@Override
				public int size() {
					return globalFilter.size();
				}

				@Override
				public float getInclusionRate() {
					return globalFilter.getInclusionRate();
				}

				@Override
				public boolean isValid(int label) {
					return globalFilter.isValid(label) && invalidIds.contains(label) == false;
				}

				@Override
				public void forEachValidId(IntConsumer action) {
					globalFilter.forEachValidId(id -> {
						if(invalidIds.contains(id) == false)
							action.accept(id);
					});
				}
			};
		}
		return deg.explore(queryIds, desiredCount, eps, filter);
	}


	
	// ---------------------------------------------------------------------------------------------------
	// ------------------------------------------- Needed to load graph ----------------------------------
	// ---------------------------------------------------------------------------------------------------

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
