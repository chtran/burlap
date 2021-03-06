package burlap.behavior.singleagent.rmax;

import java.util.List;

import javax.management.RuntimeErrorException;

import burlap.behavior.singleagent.learning.tdmethods.QLearningStateNode;
import burlap.behavior.singleagent.shaping.potential.GridWorldPotential;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class RmaxWithShaping extends Rmax {

	protected GridWorldPotential gwp;

	public void setGridWorldPotential(GridWorldPotential gwp) {
		this.gwp = gwp;
	}
	
	public RmaxWithShaping(Domain domain, RewardFunction rf,
			TerminalFunction tf, double gamma, StateHashFactory hashingFactory,
			double goalReward, int maxEpisodeSize, int m) {
		super(domain, rf, tf, gamma, hashingFactory, goalReward, maxEpisodeSize, m);
	}
	
	// Override so that initial q value is based on shaping function
	@Override
	protected QLearningStateNode getStateNode(StateHashTuple s) {
		
		QLearningStateNode node = qIndex.get(s);
		if (node == null) {
			List <GroundedAction> gas = this.getAllGroundedActions(s.s);
			if (gas.isEmpty()) {
				gas = this.getAllGroundedActions(s.s);
				throw new RuntimeErrorException(new Error(
						"No possible actions in this state, cannot continue Q-learning"));
			}
			
			node = new QLearningStateNode(s);
			for (GroundedAction ga : gas) {
				//node.addQValue(ga, gwp.potentialValue(s.s));
				node.addQValue(ga, qInitFunction.qValue(s.s, ga) - gwp.potentialValue(initialState));
			}
			qIndex.put(s, node);
		}
		
		return node;
	}

}
