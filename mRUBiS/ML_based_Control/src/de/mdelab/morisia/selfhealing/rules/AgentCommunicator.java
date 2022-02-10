package de.mdelab.morisia.selfhealing.rules;

import java.util.concurrent.atomic.AtomicBoolean;

public class AgentCommunicator {
	
	
	public static String getMessage(AtomicBoolean reset) {
		
		String fromPython = ChunkedSocketCommunicator.readln();
		if (fromPython.equals("reset")) {
			reset.set(true);
			ChunkedSocketCommunicator.println("resetting");
			return fromPython;
		}
		return fromPython;	
	}
	
	public static void waitForAgentRequestingState(AtomicBoolean reset) {
		String fromPython = "";
		while (!fromPython.equals("get_state")) {
			fromPython = getMessage(reset);
			if (reset.get()) {
				return;
			}		
		}
		return;
	}
	
	public static void waitForReset() {
		ChunkedSocketCommunicator.waitForMessage("reset");					
		ChunkedSocketCommunicator.println("resetting");
	}
}
