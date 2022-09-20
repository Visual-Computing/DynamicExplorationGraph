package com.vc.deg;

import java.util.Iterator;

/**
 * Ensures each {@link SearchEntry} in this list is unique by its {@link SearchEntry#nodeId}.
 * 
 * TODO: might be replaced by just Iterable<SearchEntry>
 * 
 * @author Nico Hezel
 */
public interface SearchResult extends Iterable<com.vc.deg.SearchResult.SearchEntry>  {
	
//	/**
//	 * Get the {@link SearchEntry} at index 
//	 *  
//	 * @param index
//	 * @return
//	 */
//	public SearchEntry get(int index)

//	public Iterable<Integer> getNodeIds();
//	public Iterable<Float> getDistances();
	
//	public Iterator<? extends SearchEntry> iterator();
	
	/**
	 * Single entry in a {@link SearchResultSet}
	 * 
	 * TODO write complete class and move it to reference, by replacing the content of ObjectDistance
	 * 
	 * @author Nico Hezel
	 */
	public static interface SearchEntry extends Comparable<SearchEntry> {
			
		public int getLabel();
		
		public float getDistance();
		
		@Override
		public boolean equals(Object obj);

		@Override
		public default int compareTo(SearchEntry o) {
			// TODO replace with Integer.compare or Comperator.comparingInt().thanComparingInt();
	        if (getDistance() == o.getDistance())
	            return Integer.compare(getLabel(), o.getLabel());
	        else
	            return Float.compare(getDistance(), o.getDistance());
		}
	}
}