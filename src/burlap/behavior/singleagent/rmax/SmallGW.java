package burlap.behavior.singleagent.rmax;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;


/*
 * Implementation of the small grid world described in 
 * Potential-based Shaping in Model-based Reinforcement Learning
 * Asmuth, Littman, Zinkov
 * Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence
 * 2008
 */
public class SmallGW {

	Domain domain;
	GridWorldDomain gwd;
	LocTF tf;
	LocRF rf;
	double discountFactor = 1.0;
	double stepCost = -0.1;
	boolean [][] northWalls;
	boolean [][] eastWalls;
	Position [] pitPos;
	Position [] goalPos;
	State 						initialState;
	DiscreteStateHashFactory	hashingFactory;
	StateParser sp;
	
	public SmallGW() {
		gwd = new GridWorldDomain(4,6); // Column, Row (x,y)
		
		northWalls = new boolean [4][6];
		for (boolean [] row : northWalls)
			Arrays.fill(row, false);
		
		northWalls[0][4] = true;
		
		gwd.setNorthWalls(northWalls);
		
		eastWalls = new boolean [4][6];
		for (boolean [] row : eastWalls)
			Arrays.fill(row, false);
		
		//eastWalls[0][4] = true;
		
		gwd.setEastWalls(eastWalls);
		
		double [][] transitionDynamics = new double [][] {
			{0.8, 0., 0.1, 0.1, 0.},
			{0., 0.8, 0.1, 0.1, 0.},
			{0.1, 0.1, 0.8, 0., 0.},
			{0.1, 0.1, 0., 0.8, 0.},
			{0., 0., 0., 0., 1.}
		};
//		double [][] transitionDynamics = new double [][] {
//				{1., 0., 0., 0.},
//				{0., 1., 0., 0.},
//				{0., 0., 1., 0.},
//				{0., 0., 0., 1.}
//			};
		gwd.setTransitionDynamics(transitionDynamics);
		domain = gwd.generateDomain();
		sp = new GridWorldStateParser(domain);
		
		// goals and pits
		goalPos = new Position [] {
			new Position(1,0),
		};
		pitPos = new Position [] {
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
		rf = new LocRF(goalPos, 5.0, pitPos, -1.0);
		
		initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
		gwd.setInitialPosition(0,5);
		GridWorldDomain.setAgent(initialState, 0, 5);
		GridWorldDomain.setLocation(initialState, 0, goalPos[0].x, goalPos[0].y);
		
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT,
				domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);
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
		//evaluateGoNorthPolicy();
		String outputPath = "/gpfs/main/home/oyakawa/Courses/2013-3-CS_2951F/Final_Project/output";
		if(!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}
		

		
		QLearning qlplanner = new QLearning(domain, rf, tf,
				discountFactor, hashingFactory, 0., .02, 10000);
		qlplanner.setMaximumEpisodesForPlanning(2000);
		qlplanner.setNumEpisodesToStore(2000);
		qlplanner.planFromState(initialState);
		List<EpisodeAnalysis> episodes = qlplanner.getAllStoredLearningEpisodes();
		System.out.println("length "+episodes.size());
		int i=1;
		for (EpisodeAnalysis ea: episodes) {
			System.out.println(i);
			i++;
			double pReturn = ea.getDiscountedReturn(discountFactor);
			try {
				PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(outputPath+"qlresults.txt", true)));
			    out.println(String.valueOf(pReturn));
			    out.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		//Policy p = new GreedyQPolicy((QComputablePlanner)qlplanner);
		//qlplanner.setLearningPolicy(p);

		
//		for (int i=0; i<200;i++) {
//			initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
//			GridWorldDomain.setAgent(initialState, 0, 5);
//			GridWorldDomain.setLocation(initialState, 0, this.goalPos[0].x, this.goalPos[0].y);
//			//qlplanner.planFromState(initialState);
//
//			EpisodeAnalysis ea = p.evaluateBehavior(initialState, rf, tf, 1000);
//			//ea.writeToFile(outputPath + "QLearningResult", sp);
//			
//			double pReturn = ea.getDiscountedReturn(discountFactor);
//			System.out.println(pReturn);
//	
//			try {
//				PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(outputPath+"qlresults.txt", true)));
//			    out.println(String.valueOf(pReturn));
//			    out.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
//			//qlplanner = new QLearning(domain, rf, tf,
//			//		discountFactor, hashingFactory, 0., .02, p, 10000);
//
//			//p = new GreedyQPolicy((QComputablePlanner)qlplanner);
//
//		}
//
//		//visualizeEpisode(outputPath);
	}
	
	public void visualizeEpisode(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(domain, gwd.getMap(), northWalls, eastWalls);
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, domain, sp, outputPath);
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
		SmallGW myWorld = new SmallGW();	
		//myWorld.visualExplorer();
		//for (int ii = 0; ii < 2000; ii++)
			myWorld.evaluatePolicy();
	}

}
