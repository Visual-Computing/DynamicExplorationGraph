package com.vc.deg.designer;

import com.vc.deg.data.DataRepository;
import com.vc.deg.graph.WeightedUndirectedGraph;

/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphDesigner<T> {
	
	protected WeightedUndirectedGraph graph;
	protected DataRepository<T> repository;

	public EvenRegularGraphDesigner(WeightedUndirectedGraph graph, DataRepository<T> repository) {
		this.graph = graph;
		this.repository = repository;
	}

	/**
	 * Add this new item to the graph
	 * 
	 * @param id
	 */
	public void add(int id) {


	}
	
	public void extendGraph(int id, boolean randomPosition) {
		if(randomPosition)
			randomGraphExtension(id);
		
		
		// TODO anns entry point should be always the same node
		
	}
	
	protected void randomGraphExtension(int id) {
		// TODO use worst edges instead of random edge
	}

}
