package com.vc.deg.ref.search;

import java.util.Iterator;
import java.util.TreeSet;

import com.vc.deg.SearchResult;

/**
 * TODO Remove class
 * @author Neiko
 *
 */
public class ResultSet implements SearchResult {
	
	protected TreeSet<? extends SearchEntry> result;
	
	public ResultSet(TreeSet<? extends SearchEntry> result) {
		this.result = result;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Iterator<SearchEntry> iterator() {
		return (Iterator<SearchEntry>) result.iterator();
	}
}
