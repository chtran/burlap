package burlap.behavior.singleagent.shaping.potential;

import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.Position;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;

public class GridWorldPotential implements PotentialFunction {
	private GridWorldDomain domain;
	private Position goal;
	private double stepCost;
	private double rho;
	private double goalValue;
	
	public GridWorldPotential(GridWorldDomain domain, Position goal, double stepCost, double rho, double goalValue) {
		this.domain = domain;
		this.goal = goal;
		this.stepCost = stepCost;
		this.rho = rho;
		this.goalValue = goalValue;
	}

	@Override
	public double potentialValue(State s) {
		ObjectInstance agent = s.getObjectsOfTrueClass(GridWorldDomain.CLASSAGENT).get(0);
		int x = agent.getDiscValForAttribute(GridWorldDomain.ATTX);
		int y = agent.getDiscValForAttribute(GridWorldDomain.ATTY);
		return stepCost * domain.getDistance(new Position(x,y), this.goal) / rho + goalValue;
	}

}
