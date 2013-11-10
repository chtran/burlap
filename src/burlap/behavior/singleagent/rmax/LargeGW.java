package burlap.behavior.singleagent.rmax;

import java.util.Arrays;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;


/*
 * Implementation of the large grid world described in 
 * Potential-based Shaping in Model-based Reinforcement Learning
 * Asmuth, Littman, Zinkov
 * Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence
 * 2008
 */
public class LargeGW {

	Domain domain;
	GridWorldDomain gwd;
	LocTF tf;
	LocRF rf;
	double discountFactor = 1.0;
	double stepCost = -0.001;
	boolean [][] northWalls;
	boolean [][] eastWalls;
	
	public LargeGW() {
		gwd = new GridWorldDomain(15,15); // Column, Row (x,y)
		for (int r = 1; r <= 4; r++) {
			//gwd.setObstacleInCell(0, r);
			//gwd.setObstacleInCell(2, r);
			//gwd.horizontalWall(0, 3, 0);
		}
		
		northWalls = new boolean [15][15];
		for (boolean [] row : northWalls)
			Arrays.fill(row, false);
		
		northWalls[0][4] = true;
		
		gwd.setNorthWalls(northWalls);
		
		eastWalls = new boolean [15][15];
		for (boolean [] row : eastWalls)
			Arrays.fill(row, false);
		
		//eastWalls[0][4] = true;
		
		gwd.setEastWalls(eastWalls);
		
//		double [][] transitionDynamics = new double [][] {
//			{0.8, 0., 0.1, 0.1},
//			{0., 0.8, 0.1, 0.1},
//			{0.1, 0.1, 0.8, 0.},
//			{0.1, 0.1, 0., 0.8}
//		};
		double [][] transitionDynamics = new double [][] {
				{1., 0., 0., 0.},
				{0., 1., 0., 0.},
				{0., 0., 1., 0.},
				{0., 0., 0., 1.}
			};
		gwd.setTransitionDynamics(transitionDynamics);
		domain = gwd.generateDomain();
		
		// goals and pits
		Position [] goalPos = new Position [] {
			new Position(1,0),
		};
		Position [] pitPos = new Position [] {
			new Position(0,1),
			new Position(0,2),
			new Position(0,3),
			new Position(0,4),
			new Position(2,1),
			new Position(2,2),
			new Position(2,3),
			new Position(2,4),
		};
		
		tf = new LocTF(goalPos, pitPos);
		rf = new LocRF(goalPos, 1.0, pitPos, -1.0);
	}

	class Position {
		int x, y;
		
		public Position(int x, int y) {
			this.x = x;
			this.y = y;
		}
		
		@Override
	    public boolean equals(Object obj) {
			boolean bothAreEqual = false;
			
	        if (!(obj instanceof Position))
	        	bothAreEqual = false;
	        else if (this == obj)
	        	bothAreEqual = true;
	        else
	        	bothAreEqual =    equal(x, ((Position) obj).x)
	                           && equal(y, ((Position) obj).y);
	        
	        return bothAreEqual;
	    }
		
	    private boolean equal(Object o1, Object o2) {
	        return o1 == null ? o2 == null : (o1 == o2 || o1.equals(o2));
	    }
	}
	
	class LocTF implements TerminalFunction {
		Position [] tPos;
		
		public LocTF(Position [] goalPos, Position [] pitPos) {
			tPos = new Position[goalPos.length + pitPos.length];
			System.arraycopy(goalPos, 0, tPos, 0, goalPos.length);
			System.arraycopy(pitPos, 0, tPos, goalPos.length, pitPos.length);
			//for (int p = 0; p < tPos.length; p++)
			//	System.out.println(tPos[p].x + " " + tPos[p].y);
		}
		
		@Override
		public boolean isTerminal(State s) {
			ObjectInstance agent = s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
			Position aPos = new Position(
			    agent.getDiscValForAttribute(GridWorldDomain.ATTX),
			    agent.getDiscValForAttribute(GridWorldDomain.ATTY));
			
			boolean foundTerminalState = false;
			for (int p = 0; p < tPos.length; p++) {
				if (tPos[p].equals(aPos)) {
					foundTerminalState = true;
					break;
				}
			}
			return foundTerminalState;
		}
	}
	
	class LocRF implements RewardFunction {
		Position [] goalPos, pitPos;
		double goalReward, pitReward;
		
		public LocRF(Position [] goalPos, double goalReward,
				     Position [] pitPos, double pitReward) {
			this.goalPos = goalPos;
			this.pitPos = pitPos;
			this.goalReward = goalReward;
			this.pitReward = pitReward;
		}
		
		@Override
		public double reward(State s, GroundedAction a, State sprime) {
			ObjectInstance agent = sprime.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
			Position aPos = new Position(
				agent.getDiscValForAttribute(GridWorldDomain.ATTX),
				agent.getDiscValForAttribute(GridWorldDomain.ATTY));

			double r = stepCost;

			boolean foundGoal = false;
			for (int p = 0; p < goalPos.length  &&  !foundGoal; p++) {
				if (goalPos[p].equals(aPos)) {
					foundGoal = true;
					r = goalReward;
				}
			}
			for (int p = 0; p < pitPos.length  &&  !foundGoal; p++) {
				if (pitPos[p].equals(aPos)) {
					foundGoal = true;
					r = pitReward;
				}
			}
			
			return r;
		}
	}
	
	public void visualExplorer(){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwd.getMap(),
				gwd.getNorthWalls(), gwd.getEastWalls());
		
		State s = GridWorldDomain.getOneAgentNLocationState(domain, 0);
		GridWorldDomain.setAgent(s, 0, 5);
		
		VisualExplorer exp = new VisualExplorer(domain, v, s);
		exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
		exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
		exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);
		exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
		
		exp.initGUI();
	}
	
	public void evaluatePolicy() {
		evaluateGoNorthPolicy();
	}
	
	public void evaluateGoNorthPolicy() {
		Policy p = new Policy() {
			@Override
			public boolean isStochastic() {
				return false;
			}
			
			@Override
			public List<ActionProb> getActionDistributionForState(State s) {
				return this.getDeterministicPolicy(s);
			}
			
			@Override
			public GroundedAction getAction(State s) {
				return new GroundedAction(domain.getAction(GridWorldDomain.ACTIONNORTH), "");
			}
		};
		
		State s = GridWorldDomain.getOneAgentNLocationState(domain, 0);
		GridWorldDomain.setAgent(s, 0, 5);
		EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf, 1000);
		double pReturn = ea.getDiscountedReturn(discountFactor);
		System.out.println(pReturn);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		LargeGW myWorld = new LargeGW();
		myWorld.visualExplorer();
		//for (int ii = 0; ii < 20; ii++)
		//	myWorld.evaluatePolicy();
	}

}