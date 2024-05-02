package org.roaringbitmap;

public class RoaringBitmapAdapter {

	public static RoaringArray getHighLowContainer(RoaringBitmap rm) {
		return rm.highLowContainer;
	}
	
	public static void move(RoaringBitmap from, RoaringBitmap to) {
		to.highLowContainer = getHighLowContainer(from);
	}
}
