package com.vs.deg.ref.graph;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;

import com.koloboke.collect.IntCursor;
import com.koloboke.collect.map.IntFloatCursor;
import com.koloboke.collect.map.IntIntMap;
import com.koloboke.collect.map.hash.HashIntIntMaps;
import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.VertexData;

public class Toy2DGraphTest {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		final int graphSize = 30;
		final int edgesPerVertex = 4;
		final Path graphInputFile = Paths.get("e:\\Data\\Feature\\2DGraph\\V"+graphSize+"_L2_K"+edgesPerVertex+"_AddK"+(edgesPerVertex*2)+"Eps0.2High_SwapK"+(edgesPerVertex*2)+"-0StepEps0.001Low.deg");
		
		final FeatureSpace space = new FloatL2Space(2);
		FeatureSpace.registerFeatureSpace(space);
		ArrayBasedWeightedUndirectedRegularGraph graph = new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, graphSize, space);
		final EvenRegularGraphDesigner graphDesigner = new EvenRegularGraphDesigner(graph);
		graphDesigner.setExtendK(edgesPerVertex*2);
		graphDesigner.setExtendEps(0.2f);
		graphDesigner.setImproveK(edgesPerVertex*2);
		graphDesigner.setImproveEps(0.001f);
		
		// build graph
		final float[][] datapoints = create2DPoints(graphSize, 100, 10);
		for (int i = 0; i < graphSize; i++) 
			graphDesigner.add(i, new FloatFeature(datapoints[i]));		
		graphDesigner.build((long step, long added, long deleted, long improved, long tries, int lastAdd, int lastDelete) -> {
			if(added == graphSize)
				graphDesigner.stop();
		});
		System.out.println("DEG has an ANR of "+graphDesigner.calcAvgNeighborRank()+" and an AEW of "+graphDesigner.calcAvgEdgeWeight());
		
		graph.writeToFile(graphInputFile);
		paintEdgeSelection(graph, replaceExtension(graphInputFile, ".png"), 1000);
	}
	

	/**
	 * Paint all nodes and their edges in an image file
	 * 
	 * @param graph
	 * @param pngFile
	 * @throws IOException 
	 */
	public static void paintEdgeSelection(ArrayBasedWeightedUndirectedRegularGraph graph, Path pngFile, int imageSize) throws IOException {
		final int kneighbors = 3;
		final int maxVertexCount = 8;
		
		class IntFloat {
			int index;
			float distance;
			public IntFloat(int index, float distance) {
				this.index = index;
				this.distance = distance;
			}
			public float getDistance() {
				return distance;
			}
			public int getIndex() {
				return index;
			}
			@Override
			public String toString() {
				return ""+index+"("+distance+")";
			}
		}
		
		final int seedId = 3;
		final FeatureVector seedFV = graph.getVertexById(seedId).getFeature();
		final FeatureSpace space = graph.getFeatureSpace();
		final IntIntMap selectedVertexIds = HashIntIntMaps.newMutableMap();
		
//		// start at radius 0 with the selected/initial image id
//		Set<Integer> newIdsAtRadius = new LinkedHashSet<>();
//		newIdsAtRadius.add(seedId);
//
//		// find the neighbors inside each radius and add them to the subgraph
//		for (int r = 0; r <= hops && newIdsAtRadius.size() > 0 && selectedVertexIds.size() < maxVertexCount; r++) {
//			final Set<Integer> newIdsAtNextRadius = new LinkedHashSet<>();
//
//			// check all neighbor ids at the current radius
//			final Iterator<Integer> vertexIds = newIdsAtRadius.iterator();
//			while(vertexIds.hasNext() && selectedVertexIds.size() < maxVertexCount) {
//				final int vertexId = vertexIds.next();
//				if(selectedVertexIds.containsKey(vertexId))
//					continue;
//				selectedVertexIds.put(vertexId, selectedVertexIds.size());
//
//				// collect all neighbors and sort them by their weights
//				final List<IntFloat> neighbors = new ArrayList<>();
//				for(Map.Entry<Integer, Float> neighbor : graph.getVertexById(vertexId).getEdges().entrySet()) 
//					neighbors.add(new IntFloat(neighbor.getKey(), neighbor.getValue()));
//				neighbors.sort(Comparator.comparingDouble(IntFloat::getDistance).thenComparingInt(IntFloat::getIndex));
//
//				// copy to subgraph
//				final int size = Math.min(kneighbors, neighbors.size());
//				for (int i = 0; i < size; i++) {
//					final int neighborId = neighbors.get(i).getIndex();
//					if(selectedVertexIds.containsKey(neighborId) == false)
//						newIdsAtNextRadius.add(neighborId);
//				}
//			}
//
//			// replace the set of ids to check with the ids of the next radius
//			newIdsAtRadius = newIdsAtNextRadius;
//		}

		final TreeSet<IntFloat> bestList = new TreeSet<>(Comparator.comparingDouble(IntFloat::getDistance).thenComparingInt(IntFloat::getIndex));
		bestList.add(new IntFloat(seedId, 0));
		selectedVertexIds.addValue(seedId, selectedVertexIds.size());
		while(bestList.size() > 0 && selectedVertexIds.size() < maxVertexCount) { 
			final int vertexId = bestList.pollFirst().getIndex();

			// collect all neighbors and sort them by their weights			
			final List<IntFloat> neighbors = new ArrayList<>();
			for(Map.Entry<Integer, Float> neighbor : graph.getVertexById(vertexId).getEdges().entrySet()) {
				final int neighborId = neighbor.getKey();
				final float dist = space.computeDistance(seedFV, graph.getVertexById(neighborId).getFeature());
				neighbors.add(new IntFloat(neighborId, dist));
			}
			neighbors.sort(Comparator.comparingDouble(IntFloat::getDistance).thenComparingInt(IntFloat::getIndex));

			// copy to vertex induced subgraph
			final int size = Math.min(kneighbors, neighbors.size());
			for (int i = 0; i < size; i++) {
				final int neighborId = neighbors.get(i).getIndex();
				if(selectedVertexIds.containsKey(neighborId) == false) {
					selectedVertexIds.put(neighborId, selectedVertexIds.size());
					bestList.add(new IntFloat(neighborId, neighbors.get(i).getDistance()));
				}
			}
		}
		
		
		
		final int border = 50;
		final int nodeSize = 40;
		final int strokeSize = 6;
		
		// find the highest value per dimension
		float[] maxPerDim = new float[2];
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			for (int dim = 0; dim < 2; dim++) 
				maxPerDim[dim] = Math.max(maxPerDim[dim], fv.readFloat(dim*Float.BYTES));
		}
		
		// create the output image
		final BufferedImage bi = new BufferedImage(imageSize+border*2, imageSize+border*2, BufferedImage.TYPE_INT_ARGB);
		final Graphics2D g2d = bi.createGraphics();
		final RenderingHints rh = new RenderingHints(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		g2d.setRenderingHints(rh);
		g2d.setStroke(new BasicStroke(strokeSize));
		
		// draw edges
		g2d.setColor(new Color(91,155,213)); // blue		
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			for(Map.Entry<Integer, Float> edge : node.getEdges().entrySet()) {
				final FeatureVector fvNeighbor = graph.getVertexById(edge.getKey()).getFeature();
				final float xEnd = fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				g2d.drawLine((int)xStart, (int)yStart, (int)xEnd, (int)yEnd);
			}
		}
		
		// draw edges distances 
	    final FontMetrics fm = g2d.getFontMetrics();
		g2d.setFont(new Font("Arial", Font.ITALIC, 20));
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			for(Map.Entry<Integer, Float> edge : node.getEdges().entrySet()) {
				final FeatureVector fvNeighbor = graph.getVertexById(edge.getKey()).getFeature();
				final float xEnd = fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				
				// compute middle of line
				final float dist = computeL2Distance(new float[] {xStart, yStart}, new float[] {xEnd, yEnd});
				final String text = ""+(int)dist;
				final Rectangle2D textBounds = fm.getStringBounds(text, g2d);
			
				final int xMiddle = (int)((Math.abs(xStart) + Math.abs(xEnd)) / 2 - textBounds.getWidth() / 2);
				final int yMiddle = (int)((Math.abs(yStart) + Math.abs(yEnd)) / 2 - textBounds.getHeight() / 2);		
				g2d.setColor(Color.black);
				g2d.drawString(text, xMiddle, yMiddle);
			}
		}
		
		// draw vertices
		g2d.setFont(new Font("Arial", Font.BOLD, 26));
		for(VertexData vertex : graph.getVertices()) {
			final FeatureVector fv = vertex.getFeature();
			final float x = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize;
			final float y = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize;
			
			if(vertex.getId() == seedId)
				g2d.setColor(Color.green);
			else if(selectedVertexIds.containsKey(vertex.getId())) {
				final float order = selectedVertexIds.get(vertex.getId()) - 1;
				final float blue = 1f - 0.6f * order / (selectedVertexIds.size() - 2);
				g2d.setColor(new Color(blue, 0, 0));
			} else
				g2d.setColor(Color.gray);
			g2d.fillOval((int)x, (int)y, nodeSize, nodeSize);

			g2d.setColor(Color.white);
			final String text = ""+vertex.getId();
			final Rectangle2D textBounds = fm.getStringBounds(text, g2d);
			g2d.drawString(text, (int)(x + textBounds.getWidth()*0.5), (int)(y + textBounds.getHeight()*1.8));
		}
		
		// store image
		ImageIO.write(bi, "png", pngFile.toFile());
	}
	
	
	
		
	/**
	 * Create 2D data points
	 * 
	 * @param count
	 * @param range
	 * @param minDist
	 * @return
	 */
	public static float[][] create2DPoints(int count, int range, int minDist) {
		final List<float[]> points = new ArrayList<>(count);
		
		final Random rnd = new Random(9);
		while(points.size() < count) {
			final float x = rnd.nextFloat() * range;
			final float y = rnd.nextFloat() * range;
			final float[] fv = new float[] {x,y};
			
			// check if the new feature vector is too close to an existing one
			float smallestDist = Float.MAX_VALUE;
			for (float[] point : points) {
				final float dist = computeL2Distance(fv, point);
				smallestDist = Math.min(dist, smallestDist);
			}
			
			if(smallestDist > minDist)
				points.add(fv);
		}
		
		return points.toArray(new float[count][]);
	}
	
	public static float computeL2Distance(float[] fv1, float[] fv2) {
		float sum = 0;
		for (int i = 0; i < fv2.length; i++) {
			final float diff = fv1[i] - fv2[i];
			sum += diff * diff;
		}
		return (float)Math.sqrt(sum);
	}
		
	public static void print2DGraph() throws IOException, ClassNotFoundException {
		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High.perfect_rng_add_only.deg");
//		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg");
//		Path graphOutputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_KCloseTo4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg");
		
		FeatureSpace space = new FloatL2Space(2);
		FeatureSpace.registerFeatureSpace(space);
		ArrayBasedWeightedUndirectedRegularGraph graph = ArrayBasedWeightedUndirectedRegularGraph.readFromFile(graphInputFile, float.class.getSimpleName());
		final EvenRegularGraphDesigner graphDesigner = new EvenRegularGraphDesigner(graph);
		System.out.println("DEG has an ANR of "+graphDesigner.calcAvgNeighborRank()+" and an AEW of "+graphDesigner.calcAvgEdgeWeight());
		
//		printGraph(graph);
		//paintGraph(graph, replaceExtension(graphInputFile, ".png"), 1000);
		
		final List<FeatureVector> points = extractDataPoints(graph);
		final ArrayBasedWeightedUndirectedRegularGraph rng = buildRNG(points, space); 
		final EvenRegularGraphDesigner rngDesigner = new EvenRegularGraphDesigner(rng);
		System.out.println("RNG has an ANR of "+rngDesigner.calcAvgNeighborRank()+" and an AEW of "+rngDesigner.calcAvgEdgeWeight());
		rng.writeToFile(graphInputFile.getParent().resolve("L2_rng.deg"));
		paintGraph(rng, graphInputFile.getParent().resolve("L2_rng.png"), 1000);
		
		final ArrayBasedWeightedUndirectedRegularGraph dg = buildDG(points, space); 
		final EvenRegularGraphDesigner dgDesigner = new EvenRegularGraphDesigner(dg);
		System.out.println("DG has an ANR of "+dgDesigner.calcAvgNeighborRank()+" and an AEW of "+dgDesigner.calcAvgEdgeWeight());
		dg.writeToFile(graphInputFile.getParent().resolve("L2_dg.deg"));
		paintGraph(dg, graphInputFile.getParent().resolve("L2_dg.png"), 1000);
		
		final ArrayBasedWeightedUndirectedRegularGraph knng = buildKNNG(points, space, 3); 
		final EvenRegularGraphDesigner knngDesigner = new EvenRegularGraphDesigner(knng);
		System.out.println("KNNG has an ANR of "+knngDesigner.calcAvgNeighborRank()+" and an AEW of "+knngDesigner.calcAvgEdgeWeight());
		knng.writeToFile(graphInputFile.getParent().resolve("L2_K3_knng.deg"));
		paintGraph(knng, graphInputFile.getParent().resolve("L2_K3_knng.png"), 1000);
		
		final ArrayBasedWeightedUndirectedRegularGraph knnAproxRNG = buildKNNGwithApproximateRNG(points, space, 3); 
		final EvenRegularGraphDesigner knngAproxRNGDesigner = new EvenRegularGraphDesigner(knnAproxRNG);
		System.out.println("KNNG+aproxRNG has an ANR of "+knngAproxRNGDesigner.calcAvgNeighborRank()+" and an AEW of "+knngAproxRNGDesigner.calcAvgEdgeWeight());
		knnAproxRNG.writeToFile(graphInputFile.getParent().resolve("L2_K3_knnAproxRNG.deg"));
		paintGraph(knnAproxRNG, graphInputFile.getParent().resolve("L2_K3_knnAproxRNG.png"), 1000);

//		// slide 8
//		changesEdgesOfVertex(graph, 0, 3,6);
//		changesEdgesOfVertex(graph, 1, 2,10);
//		changesEdgesOfVertex(graph, 2, 1,7);
//		changesEdgesOfVertex(graph, 3, 0,7,10,13);
//		changesEdgesOfVertex(graph, 4, 5,6,9,11);
//		changesEdgesOfVertex(graph, 5, 4,12);
//		changesEdgesOfVertex(graph, 6, 0,4,8,11);
//		changesEdgesOfVertex(graph, 7, 2,3,11,12);
//		changesEdgesOfVertex(graph, 8, 6,9);
//		changesEdgesOfVertex(graph, 9, 4,8);
//		changesEdgesOfVertex(graph, 10, 1,3);
//		changesEdgesOfVertex(graph, 11, 4,6,7,13);
//		changesEdgesOfVertex(graph, 12, 5,7);
//		changesEdgesOfVertex(graph, 13, 3,11);
//		
//		// slide 9
//		changesEdgesOfVertex(graph, 0, 3,6);
//		changesEdgesOfVertex(graph, 1, 2,10);
//		changesEdgesOfVertex(graph, 2, 1,5,6,7);
//		changesEdgesOfVertex(graph, 3, 0,7,10,13);
//		changesEdgesOfVertex(graph, 4, 5,9);
//		changesEdgesOfVertex(graph, 5, 2,4,11,12);
//		changesEdgesOfVertex(graph, 6, 0,2,8,11);
//		changesEdgesOfVertex(graph, 7, 2,3,11,12);
//		changesEdgesOfVertex(graph, 8, 6,9);
//		changesEdgesOfVertex(graph, 9, 4,8);
//		changesEdgesOfVertex(graph, 10, 1,3);
//		changesEdgesOfVertex(graph, 11, 5,6,7,13);
//		changesEdgesOfVertex(graph, 12, 5,7);
//		changesEdgesOfVertex(graph, 13, 3,11);
//		
//		
//		graph.writeToFile(graphOutputFile);
	}

	private static ArrayBasedWeightedUndirectedRegularGraph buildDG(List<FeatureVector> points, FeatureSpace space) {
		final ArrayBasedWeightedUndirectedRegularGraph graph = new ArrayBasedWeightedUndirectedRegularGraph(points.size(), space);
				
		// create the vertices
		for (int i = 0; i < points.size(); i++)
			graph.addVertex(i, points.get(i));
		
		// add the edges
		for (int v1Index = 0; v1Index < points.size(); v1Index++) {
			final FeatureVector v1Feature = points.get(v1Index);			
			for (int v2Index = v1Index+1; v2Index < points.size(); v2Index++) {
				final FeatureVector v2Feature = points.get(v2Index);

				// check if the edge (v1,v2) does already exists 
				if(graph.hasEdge(v1Index, v2Index) == false) {
					
					// compute center position between v1 and v2
					final float[] center = new float[2];
					center[0] = (v1Feature.readFloat(0) + v2Feature.readFloat(0)) / 2;
					center[1] = (v1Feature.readFloat(4) + v2Feature.readFloat(4)) / 2;
					final FeatureVector centerFeature = new FloatFeature(center);
					
					// radius in which no other vertex except v1 and v2 is allowed
					final float radius = space.computeDistance(centerFeature, v1Feature);
					boolean isDG = true;
					for (int v3Index = 0; v3Index < points.size(); v3Index++) {
						if(v3Index != v1Index && v3Index != v2Index) {
							final FeatureVector v3Feature = points.get(v3Index);
							
							if(space.computeDistance(centerFeature, v3Feature) < radius) {
								isDG = false;
								break;
							}
						}
					}

					// is a valid edge add to the graph
					if(isDG) 
						graph.addUndirectedEdge(v1Index, v2Index, space.computeDistance(v1Feature, v2Feature));
				}
			}
		}
		
		
		return graph;
	}
	
	private static ArrayBasedWeightedUndirectedRegularGraph buildKNNG(List<FeatureVector> points, FeatureSpace space, int k) {
		final ArrayBasedWeightedUndirectedRegularGraph graph = new ArrayBasedWeightedUndirectedRegularGraph(points.size(), space);
		
		// create the vertices
		int size = points.size();
		for (int i = 0; i < size; i++)
			graph.addVertex(i, points.get(i));
		
		class IntFloat {
			int index;
			float distance;
			public IntFloat(int index, float distance) {
				this.index = index;
				this.distance = distance;
			}
		}
		
		for (int v1Index = 0; v1Index < size; v1Index++) {
			final FeatureVector v1Feature = points.get(v1Index);
			
			// find the best vertices
			final List<IntFloat> topList = new ArrayList<>(size);
			for (int v2Index = 0; v2Index < points.size(); v2Index++) {
				final FeatureVector v2Feature = points.get(v2Index);
				topList.add(new IntFloat(v2Index, space.computeDistance(v1Feature, v2Feature)));
			}
			topList.sort((o1, o2) -> Float.compare(o1.distance, o2.distance));
			
			// add the best vertices as neighbors
			for (int i = 0; i < k; i++) {
				final IntFloat best = topList.get(i+1); //skip the self reference 
				if(graph.hasEdge(v1Index, best.index) == false)
					graph.addUndirectedEdge(v1Index, best.index, best.distance);
			}
		}
		
		return graph;
	}
		
	
	private static ArrayBasedWeightedUndirectedRegularGraph buildKNNGwithApproximateRNG(List<FeatureVector> points, FeatureSpace space, int k) {
		final ArrayBasedWeightedUndirectedRegularGraph graph = buildKNNG(points, space, k);
		
		// check all vertices
		for (int v1Index = 0; v1Index < points.size(); v1Index++) {
			final VertexData v1 = graph.getVertexByLabel(v1Index);
			
			// check only the neighbors
			final IntFloatCursor edgesCursor = v1.getEdges().cursor();
			while(edgesCursor.moveNext()) {
				if(checkRNG(graph, v1Index, edgesCursor.key(), edgesCursor.value()) == false)
					edgesCursor.remove();
			}
		}
		return graph;
	}
	
	private static boolean checkRNG(ArrayBasedWeightedUndirectedRegularGraph graph, int vertexId, int targetId, float vertexTargetWeight) {
		for(Map.Entry<Integer, Float> neighbor : graph.getVertexById(vertexId).getEdges().entrySet()) {
			float neighborTargetWeight = graph.getEdgeWeight(neighbor.getKey(), targetId);
			if(neighborTargetWeight >= 0 && vertexTargetWeight > Math.max(neighbor.getValue(), neighborTargetWeight)) 
				return false;
		}
		return true;
	}
		
	
	private static ArrayBasedWeightedUndirectedRegularGraph buildRNG(List<FeatureVector> points, FeatureSpace space) {
		final ArrayBasedWeightedUndirectedRegularGraph graph = new ArrayBasedWeightedUndirectedRegularGraph(points.size(), space);
		
		// create the vertices
		for (int i = 0; i < points.size(); i++)
			graph.addVertex(i, points.get(i));
		
		// add the edges
		for (int v1Index = 0; v1Index < points.size(); v1Index++) {
			final FeatureVector v1Feature = points.get(v1Index);			
			for (int v2Index = v1Index+1; v2Index < points.size(); v2Index++) {
				final FeatureVector v2Feature = points.get(v2Index);

				// check if the edge (v1,v2) does already exists 
				if(graph.hasEdge(v1Index, v2Index) == false) {					
					final float distV1V2 = space.computeDistance(v1Feature, v2Feature);
					
					// is (v1,v2) a valid RNG edge?
					boolean isRNG = true;
					for (int v3Index = 0; v3Index < points.size(); v3Index++) {

						// try to find another vertex which is inside the lune of v1 and v2
						if(v3Index != v1Index && v3Index != v2Index) {
							
							final FeatureVector v3Feature = points.get(v3Index);
							final float distV1V3 = space.computeDistance(v1Feature, v3Feature);
							final float distV2V3 = space.computeDistance(v2Feature, v3Feature);
							
							if(distV1V2 > distV1V3 && distV1V2 > distV2V3) {
								isRNG = false;
								break;
							}
						}
					}
					
					if(isRNG) 
						graph.addUndirectedEdge(v1Index, v2Index, distV1V2);
				}				
			}
		}
		
		return graph;
	}

	private static List<FeatureVector> extractDataPoints(ArrayBasedWeightedUndirectedRegularGraph graph) {
		final List<FeatureVector> dataPoints = new ArrayList<>();
		for(VertexData vertex : graph.getVertices()) {
			dataPoints.add(vertex.getFeature());
		}
		return dataPoints;
	}

	/**
	 * Replaces the edges of the vertex with new neighbors
	 * 
	 * @param graph
	 * @param vertexIdx
	 * @param neighborIndices
	 */
	public static void changesEdgesOfVertex(ArrayBasedWeightedUndirectedRegularGraph graph, int vertexIdx, int ... neighborIndices) {
		VertexData data = graph.getVertexById(vertexIdx);
		Map<Integer,Float> edges = data.getEdges();
		edges.clear();
		
		for (int neigborIndex : neighborIndices) {
			VertexData neighbor = graph.getVertexById(neigborIndex);
			edges.put(neigborIndex, graph.getFeatureSpace().computeDistance(data.getFeature(), neighbor.getFeature()));
		}		
	}	
	
	
	/**
	 * Paint all nodes and their edges in an image file
	 * 
	 * @param graph
	 * @param pngFile
	 * @throws IOException 
	 */
	public static void paintGraph(ArrayBasedWeightedUndirectedRegularGraph graph, Path pngFile, int imageSize) throws IOException {
		final int border = 50;
		final int nodeSize = 40;
		final int strokeSize = 6;
		
		// find the highest value per dimension
		float[] maxPerDim = new float[2];
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			for (int dim = 0; dim < 2; dim++) 
				maxPerDim[dim] = Math.max(maxPerDim[dim], fv.readFloat(dim*Float.BYTES));
		}
		
		// create the output image
		final BufferedImage bi = new BufferedImage(imageSize+border*2, imageSize+border*2, BufferedImage.TYPE_INT_ARGB);
		final Graphics2D g2d = bi.createGraphics();
		final RenderingHints rh = new RenderingHints(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		g2d.setRenderingHints(rh);
		g2d.setStroke(new BasicStroke(strokeSize));
		
		// draw edges
		g2d.setColor(new Color(91,155,213)); // blue		
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			for(Map.Entry<Integer, Float> edge : node.getEdges().entrySet()) {
				final FeatureVector fvNeighbor = graph.getVertexById(edge.getKey()).getFeature();
				final float xEnd = fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				g2d.drawLine((int)xStart, (int)yStart, (int)xEnd, (int)yEnd);
			}
		}
		
		// draw edges distances 
	    final FontMetrics fm = g2d.getFontMetrics();
		g2d.setFont(new Font("Arial", Font.ITALIC, 20));
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			for(Map.Entry<Integer, Float> edge : node.getEdges().entrySet()) {
				final FeatureVector fvNeighbor = graph.getVertexById(edge.getKey()).getFeature();
				final float xEnd = fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				
				// compute middle of line
				final float dist = computeL2Distance(new float[] {xStart, yStart}, new float[] {xEnd, yEnd});
				final String text = ""+(int)dist;
				final Rectangle2D textBounds = fm.getStringBounds(text, g2d);
			
				final int xMiddle = (int)((Math.abs(xStart) + Math.abs(xEnd)) / 2 - textBounds.getWidth() / 2);
				final int yMiddle = (int)((Math.abs(yStart) + Math.abs(yEnd)) / 2 - textBounds.getHeight() / 2);		
				g2d.setColor(Color.black);
				g2d.drawString(text, xMiddle, yMiddle);
			}
		}
		
		// draw vertices
		g2d.setFont(new Font("Arial", Font.BOLD, 26));
		for(VertexData vertex : graph.getVertices()) {
			final FeatureVector fv = vertex.getFeature();
			final float x = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize;
			final float y = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize;			
			g2d.setColor(Color.gray);
			g2d.fillOval((int)x, (int)y, nodeSize, nodeSize);

			g2d.setColor(Color.white);
			final String text = ""+vertex.getId();
			final Rectangle2D textBounds = fm.getStringBounds(text, g2d);
			g2d.drawString(text, (int)(x + textBounds.getWidth()*0.5), (int)(y + textBounds.getHeight()*1.8));
		}
		
		// store image
		ImageIO.write(bi, "png", pngFile.toFile());
	}
	
	/**
	 * Print all nodes and their edges on the console
	 * 
	 * @param graph
	 */
	public static void printGraph(ArrayBasedWeightedUndirectedRegularGraph graph) {
		for(VertexData data : graph.getVertices()) {		
			Map<Integer,Float> edges = data.getEdges();
			System.out.print("Neighbors of vertex "+data.getId()+": ");
			for (int neigborIndex : edges.keySet()) 
				System.out.print(neigborIndex+", ");
			System.out.println();
		}
	}
	
	/**
	 * Changes the extension of a file
	 * 
	 * @param file
	 * @param newExtension
	 * @return
	 */
	public static Path replaceExtension(Path file, String newExtension) {
		String fileString = file.toString();
		int end = fileString.lastIndexOf('.');
		return Paths.get(fileString.subSequence(0, end) + newExtension);
	}
	
	public static class FloatL2Space implements FeatureSpace {

		protected int dims;
		
		public FloatL2Space(int dims) {
			this.dims = dims;
		}
		
		@Override
		public int featureSize() {
			return dims*Float.BYTES;
		}
		
		@Override
		public int dims() {
			return dims;
		}
		
		@Override
		public Class<?> getComponentType() {
			return float.class;
		}

		@Override
		public int metric() {
			return Metric.L2.getId();
		}
		
		@Override
		public boolean isNative() {
			return false;
		}
		
		@Override
		public float computeDistance(FeatureVector f1, FeatureVector f2) {
			final int byteSize = dims * Float.BYTES;
			
			float result = 0;
			for (int i = 0; i < byteSize; i+=Float.BYTES) {
				float diff = f1.readFloat(i) - f2.readFloat(i);
				result += diff*diff;
			}
			
			return result;
		}
	}
	
	
//	protected static class IntFloat {
//		private final int index;
//		private final float distance;
//	
//		public IntFloat(int index, float distance) {
//			this.index = index;
//			this.distance = distance;
//		}
//	
//		public int getIndex() {
//			return index;
//		}
//	
//		public float getDistance() {
//			return distance;
//		}
//		
//		@Override
//		public String toString() {
//			return "index: "+index+", distance: "+distance;
//		}
//	
//		@Override
//		public boolean equals(Object obj) {
//			if(obj instanceof IntFloat)
//				return ((IntFloat) obj).getIndex() == index;
//			return super.equals(obj);
//		}
//		
//		/**
//		 * Sort the distances from lowest to highest, followed by the index
//		 *
//		 * @return
//		 */
//		public static Comparator<IntFloat> asc() {
//			return Comparator.comparingDouble(IntFloat::getDistance).thenComparingInt(IntFloat::getIndex);
//		}
//	}
}
