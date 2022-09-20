package com.vc.deg.ref.navigation;

import com.vc.deg.GraphNavigator;
import com.vc.deg.graph.WeightedUndirectedGraph;
import com.vc.deg.ref.graph.MapBasedWeightedUndirectedRegularGraph;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class MapBasedGraphNavigator implements GraphNavigator {

	protected MapBasedWeightedUndirectedRegularGraph graph;

	public MapBasedGraphNavigator(MapBasedWeightedUndirectedRegularGraph graph) {
		this.graph = graph;
	}

}
