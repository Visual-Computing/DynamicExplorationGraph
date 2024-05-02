package org.roaringbitmap;

public class RoaringBitmapAdapter {

	public static RoaringArray getHighLowContainer(RoaringBitmap rm) {
		return rm.highLowContainer;
	}
	
	public static void move(RoaringArray obj, RoaringBitmap to) {
		to.highLowContainer = obj;
	}
	
	public static void move(RoaringBitmap from, RoaringBitmap to) {
		move(getHighLowContainer(from), to);
	}
}
