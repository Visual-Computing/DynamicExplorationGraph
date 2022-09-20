package com.vc.deg.ref.hierarchy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class UniformDistributedRankSet {

	protected static final float desiredShrinkFactor = 4;
	
	protected final Random rnd = new Random(7);

	protected final int topRankSize;
	protected final List<Set<Integer>> rankToKey;
	protected final Map<Integer, Integer> keyToRank;
	
	
	public UniformDistributedRankSet(int topRankSize) {		
		this.topRankSize = topRankSize;
		this.keyToRank = new HashMap<>();
		this.rankToKey = new ArrayList<>();
		ensureRank(0);
	}
	
	/***
	 * Copy constructor
	 * @param topRankSize
	 * @param rnd
	 */
	protected UniformDistributedRankSet(int topRankSize, List<Set<Integer>> rankToKey, Map<Integer, Integer> keyToRank) {		
		this.topRankSize = topRankSize;
		this.keyToRank = keyToRank;
		this.rankToKey = rankToKey;
		ensureRank(0);
	}

	// ----------------------------------------------------------------------------------------------------------
	// ------------------------------------------- Graph Rank Methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	protected int getRandomKey(int atRank) {
		final Set<Integer> keys = rankToKey.get(atRank);
		final int steps = rnd.nextInt(keys.size());
		final Iterator<Integer> it = keys.iterator();
		Integer val = it.next();
		for (int i = 0; i < steps; i++) 
			val = it.next();
		return val;
	}
	
	
	/**
	 * Ensures there is enough ranks.
	 * 
	 * @param rank
	 */
	protected void ensureRank(int rankCount) {
		
		// not enough ranks yet
		while(rankToKey.size() < rankCount) {
			
			// add another rank at the bottom (new rank 0) and fill it with the keys from the rank above
			rankToKey.add(0, new HashSet<>((rankToKey.size() == 0) ? Collections.emptyList() : rankToKey.get(0)));
						
			// promote all existing elements one rank up
			for(Map.Entry<Integer, Integer> entry : keyToRank.entrySet()) 
				entry.setValue(entry.getValue() + 1);
		}
	}
	
	/**
	 * Ensures there is enough space and ranks. 
	 * Adds a rank if needed and promotes existing elements.
	 * 
	 * @param size
	 */
	protected void ensureSize(int size) {
		
		// how many elements are at each rank
		int[] rankSizes = rankDistribution(size, topRankSize, desiredShrinkFactor);
		
		// not enough ranks yet	
		ensureRank(rankSizes.length);
	}
	
	/**
	 * Compute the rank distribution for a specific amount of elements.
	 * The max grow factor decides how much more elements are on a rank
	 * above. While the top rank has a maximum size as well.
	 * 
	 * All ranks from top to bottom have a grow factor equal to the max
	 * grow factor. Only rank 0 has a smaller factor, since it is always
	 * the given size.
	 * 
	 * @param size
	 * @param topRankSize
	 * @param maxGrowFactor
	 * @return
	 */
	protected static int[] rankDistribution(int size, int topRankSize, float maxGrowFactor) {
		
		// how many ranks are needed
		int desiredRanks = (int) Math.max(1, Math.ceil(Math.log(size / (float)topRankSize) / Math.log(maxGrowFactor)) + 1);

		// element count at each rank (with a fix top rank and all elements at rank 0)
		int[] rankSize = new int[desiredRanks];
		rankSize[0] = size;	
		rankSize[desiredRanks-1] = topRankSize;
		for (int i = desiredRanks-2; i > 0; i--) {
			rankSize[i] = (int)(rankSize[i+1] * maxGrowFactor);
		}
		
		return rankSize;
	}

	/**
	 * Add a new key at a given rank.
	 * Degrades an existing key from this rank.
	 * 
	 * @param key
	 * @param atRank
	 * @return
	 */
	protected void addAtRank(int key, int atRank) {
		
		// add key to the given rank and all ranks below
		for (int r = atRank; r > 0; r--) {
			
			// degrade random key at rank
			int keyToDegrade = getRandomKey(r);
			rankToKey.get(r).remove(keyToDegrade);
			keyToRank.put(keyToDegrade, r - 1);

			// add new key at rank
			rankToKey.get(r).add(key);
		}
		
		// add new key to rank 0 and map of all keys
		rankToKey.get(0).add(key);
		keyToRank.put(key, atRank);
	}
	
	/**
	 * Balances the ranks if a element has been removed
	 */
	protected void balanceRanks() {

		// how many elements are at each rank
		final int[] rankSizes = rankDistribution(size(), topRankSize, desiredShrinkFactor);

		// promote entire rank 0 to rank 1 and remove rank 0 afterwards
		while(rankSizes.length < rankToKey.size()) {
			final Set<Integer> rank1 = rankToKey.get(1);
			
			// promote entire rank 0 to rank 1
			final Iterator<Integer> keyIterator = rankToKey.get(0).iterator();
			while(keyIterator.hasNext()) {
				final int value = keyIterator.next();
				keyToRank.put(value, 1);
				rank1.add(value);
			}
			
			// remove rank 0 
			rankToKey.remove(0);
			
			// degrade all existing elements one rank down
			for(Map.Entry<Integer, Integer> entry : keyToRank.entrySet()) 
				entry.setValue(entry.getValue() - 1);
		}
		
		// promote ids to higher graph ranks if needed
		for (int rank = 1; rank < rankSizes.length; rank++) {			
			final Set<Integer> ids = rankToKey.get(rank);	
			final Set<Integer> idsBelow = rankToKey.get(rank-1);
			
			// fill until the current rank is big enough
			int desiredSize = rankSizes[rank];
			while(ids.size() < desiredSize) {

				// get random node from a rank below which does not exist at the current rank
				int id = -1;
				do {
					final int steps = rnd.nextInt(idsBelow.size());
					final Iterator<Integer> it = idsBelow.iterator();				
					for (int i = 0; i < steps; i++) 
						it.next();
					id = it.next();
					
					// test the next element if the random id exists on this rank 
					while(ids.contains(id) == false && it.hasNext())
						id = it.next();
					
				} while(ids.contains(id) == false);

				// update the sets
				ids.add(id);
				keyToRank.put(id, rank);

				// should not happen
				if(id == -1)
					System.out.println("No valid id to promote from " + rank + "(" + ids.size() + ") to " + (rank + 1));
			}			
		}
		
	}
	
	// -----------------------------------------------------------------

	public int maxRank() {
		return rankToKey.size() - 1;
	}

	public boolean add(int key) {
		if(keyToRank.containsKey(key) == false) {
			int newSize = size() + 1;
			
			// make sure there is enough space to add another element
			ensureSize(newSize);
			
			// compute chance to upgrade the new element at a higher rank
			int rank = 0;
			for (int r = maxRank(); r > 0; r--) {
				if(rnd.nextFloat() < ((float)rankToKey.get(r).size() / newSize)) {
					rank = r;
					break;
				}
			}
			
			// add the new element at the rank
			addAtRank(key, rank);
			
			return true;
		}
		return false;
	}
	
	public boolean remove(int key) {
		if(keyToRank.containsKey(key) == true) {
			
			// remove key from all ranks
			int rank = keyToRank.remove(key);
			for (int r = 0; r <= rank; r++) 
				rankToKey.get(r).remove(key);
			
			// balance the ranks
			balanceRanks();
			
			return true;
		}
		return false;
	}
	
	public int rank(int key) {
		return keyToRank.get(key);
	}

	public int size() {
		return keyToRank.size();
	}

	public Map<Integer, Integer> getRankMap() {
		return keyToRank;
	}
	
	public Set<Integer> getKeysAtRank(int atRank) {
		return rankToKey.get(atRank);
	}
	
	@Override
	public String toString() {
		String rankStr = "";
		for (int i = 0; i < rankToKey.size(); i++) 
			rankStr += "Rank "+i+" "+rankToKey.get(i).size()+" elements." + System.lineSeparator();
		return rankStr + System.lineSeparator() + super.toString();
	}

	public void write(Output output) {
		output.writeInt(this.topRankSize);
		output.writeInt(this.rankToKey.size());
		for (IntSet intSet : rankToKey) {
			HashIntSetSerializer.store(output, intSet);
		}
		HashIntIntMapSerializer.store(output, this.keyToRank);
	}

	public static UniformDistributedRankSet read(Input input) {
		int topRankSize = input.readInt();
		int ranks = input.readInt();
		List<IntSet> rankToKey = new ArrayList<>(ranks);
		for (int i = 0; i < ranks; i++) {
			rankToKey.add(HashIntSetSerializer.load(input));
		}
		IntIntMap keyToRank = HashIntIntMapSerializer.load(input);
		return new UniformDistributedRankSet(topRankSize, rankToKey, keyToRank);
	}
	
	@Override
	public UniformDistributedRankSet clone() throws CloneNotSupportedException {

		IntIntMap keyToRank = HashIntIntMaps.newMutableMap(this.keyToRank);
		List<IntSet> rankToKey = new ArrayList<>(this.rankToKey.size());
		for (IntSet intSet : this.rankToKey) {
			rankToKey.add(HashIntSets.newMutableSet(intSet));
		}
		return new UniformDistributedRankSet(topRankSize, rankToKey, keyToRank);
	}
}
