package com.vc.deg.viz.filter;

import java.util.function.Consumer;
import java.util.function.IntConsumer;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;

/**
 * A prepared filter provides a list of valid ids.
 * 
 * @author Nico Hezel
 */
public class PreparedGraphFilter implements MutableGraphFilter {
	
	protected final IntSet validIds;
	protected final int allElementCount;
	
	public PreparedGraphFilter(IntSet validIds, int allElementCount) {
		this.validIds = validIds;
		this.allElementCount = allElementCount;
	}
	
	@Override
	public void forEachValidId(IntConsumer action) {
		validIds.forEach(action);
	}
	
	@Override
	public PreparedGraphFilter remove(Consumer<IntConsumer> idProvider) {
		final IntSet copy = HashIntSets.newMutableSet(validIds); 
		final IntConsumer removeFunc = id -> copy.removeInt(id);
		idProvider.accept(removeFunc);
		return new PreparedGraphFilter(copy, allElementCount);
	}
	
	@Override
	public boolean isValid(int id) {
		return validIds.contains(id);
	}
	
	@Override
	public int size() {
		return validIds.size();
	}

	@Override
	public float getInclusionRate() {
		return Math.max(0, Math.min(1, ((float)size()) / allElementCount));
	}
	
	@Override
	public String toString() {
		return this.getClass().getSimpleName() + " with "+getInclusionRate()+" valid ids.";
	}
	
	
	
//	/**
//	 * Intersect of two filter results
//	 * 
//	 * @param other
//	 * @return
//	 */
//	public PreparedGraphFilter and(PreparedGraphFilter other) {
//		Objects.requireNonNull(other);
//
//		// ---- INTERSECT ----
//		final IntSet v1 = (other.size() < size()) ? other.validIds : validIds;
//		final IntSet v2 = (other.size() < size()) ? validIds : other.validIds;
//		final IntSet validIds = HashIntSets.newMutableSet(v1);		
//		validIds.removeIf((int val) -> v2.contains(val) == false);
//		return new PreparedGraphFilter(validIds);
//	}
//	
//	public GraphFilter and(GraphFilter other) {
//		if(other instanceof PreparedGraphFilter)
//			return and((PreparedGraphFilter)other);
//		
//		return new GraphFilter() {
//			
//			@Override
//			public int size() {
//				return Math.min(other.size(), size());
//			}
//			
//			@Override
//			public boolean isValid(int label) {
//				return other.isValid(label) && isValid(label);
//			}
//		};
//	}
//	
//	
//	/**
//	 * Union of two filter results
//	 * 
//	 * @param other
//	 * @return
//	 */
//	public PreparedGraphFilter or(PreparedGraphFilter other) {
//		Objects.requireNonNull(other);
//		
//		final IntSet validIds = HashIntSets.newMutableSet(other.validIds);
//		validIds.addAll(validIds);
//		
//		return new PreparedGraphFilter(validIds);
//	}
//	
//	public GraphFilter or(GraphFilter other) {
//		if(other instanceof PreparedGraphFilter)
//			return or((PreparedGraphFilter)other);
//		
//		return new GraphFilter() {
//			
//			@Override
//			public int size() {
//				return Math.max(other.size(), size());
//			}
//			
//			@Override
//			public boolean isValid(int label) {
//				return other.isValid(label) || isValid(label);
//			}
//		};
//	}
}
