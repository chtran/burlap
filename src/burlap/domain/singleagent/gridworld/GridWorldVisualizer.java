package burlap.domain.singleagent.gridworld;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;

import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.visualizer.ObjectPainter;
import burlap.oomdp.visualizer.StaticPainter;
import burlap.oomdp.visualizer.Visualizer;


public class GridWorldVisualizer {

	
	
	public static Visualizer getVisualizer(Domain d, int [][] map,
									       boolean [][] northWalls, boolean [][] eastWalls){
		
		Visualizer v = new Visualizer();
		
		v.addStaticPainter(new MapPainter(d, map, northWalls, eastWalls));
		v.addObjectClassPainter(GridWorldDomain.CLASSLOCATION, new CellPainter(Color.blue, map));
		v.addObjectClassPainter(GridWorldDomain.CLASSAGENT, new CellPainter(Color.red, map));
		
		return v;
	}
	
	
	
	public static class MapPainter implements StaticPainter{

		protected int 				dwidth;
		protected int 				dheight;
		protected int [][] 			map;
		protected boolean [][]		northWalls;
		protected boolean [][]		eastWalls;
		
		public MapPainter(Domain domain, int [][] map,
				          boolean [][] northWalls, boolean [][] eastWalls) {
			this.dwidth = map.length;
			this.dheight = map[0].length;
			this.map = map;
			this.northWalls = northWalls;
			this.eastWalls = eastWalls;
		}

		@Override
		public void paint(Graphics2D g2, State s, float cWidth, float cHeight) {
			
			//draw the walls; make them black
			g2.setColor(Color.black);
			
			float domainXScale = this.dwidth;
			float domainYScale = this.dheight;
			
			//determine then normalized width
			float width = (1.0f / domainXScale) * cWidth;
			float height = (1.0f / domainYScale) * cHeight;
			
			//pass through each cell of the map and if it is a wall, draw it
			for(int i = 0; i < this.dwidth; i++){
				for(int j = 0; j < this.dheight; j++){
					
					if(this.map[i][j] == 1){
					
						float rx = i*width;
						float ry = cHeight - height - j*height;
					
						g2.fill(new Rectangle2D.Float(rx, ry, width, height));
						
					}
					
				}
			}
			
			// Draw directional walls.
			// They will be 1/4 the size of a grid and positioned inbetween the
			// grids that they are preventing access to
			g2.setColor(Color.green);
			for (int i = 0; i < this.dwidth; i++) {
				for (int j = 0; j < this.dheight; j++) {
					if (this.northWalls[i][j]) {
						float rx = i*width;
						float ry = cHeight - height - j*height - height/8.0f;
						g2.fill(new Rectangle2D.Float(rx, ry, width, height/4.0f));
					}
					if (this.eastWalls[i][j]) {
						float rx = (i+1)*width - width/8.0f;
						float ry = cHeight - height - j*height;
						g2.fill(new Rectangle2D.Float(rx, ry, width/4.0f, height));
					}
				}	
			}
			
		}
		
		
	}
	
	
	
	public static class CellPainter implements ObjectPainter{

		protected Color			col;
		protected int			dwidth;
		protected int			dheight;
		protected int [][]		map;
		
		public CellPainter(Color col, int [][] map) {
			this.col = col;
			this.dwidth = map.length;
			this.dheight = map[0].length;
			this.map = map;
		}

		@Override
		public void paintObject(Graphics2D g2, State s, ObjectInstance ob, float cWidth, float cHeight) {
			
			
			//set the color of the object
			g2.setColor(this.col);
			
			float domainXScale = this.dwidth;
			float domainYScale = this.dheight;
			
			//determine then normalized width
			float width = (1.0f / domainXScale) * cWidth;
			float height = (1.0f / domainYScale) * cHeight;
			
			float rx = ob.getDiscValForAttribute(GridWorldDomain.ATTX)*width;
			float ry = cHeight - height - ob.getDiscValForAttribute(GridWorldDomain.ATTY)*height;
			
			g2.fill(new Rectangle2D.Float(rx, ry, width, height));
			
		}
		
		
		
		
	}
	
	
}
