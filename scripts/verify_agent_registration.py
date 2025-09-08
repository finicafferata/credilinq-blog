#!/usr/bin/env python3
"""
Agent Registration Verification Script
Checks which agents are properly registered in the AgentFactory and available for workflow execution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.core.agent_factory import AgentFactory
from src.agents.core.base_agent import AgentType

def main():
    print("🔍 Verifying Agent Registration for Workflow Execution")
    print("=" * 60)
    
    # Initialize agent factory
    try:
        factory = AgentFactory()
        print("✅ AgentFactory initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AgentFactory: {e}")
        return
    
    # Expected agents for workflow execution
    expected_agents = {
        "planner": AgentType.PLANNER,
        "researcher": AgentType.RESEARCHER, 
        "writer": AgentType.WRITER,
        "editor": AgentType.EDITOR,
        "seo": AgentType.SEO,
        "image": AgentType.IMAGE_PROMPT,
        "social_media": AgentType.SOCIAL_MEDIA,
        "campaign_manager": AgentType.CAMPAIGN_MANAGER,
        "content_repurposer": AgentType.CONTENT_REPURPOSER,
    }
    
    print(f"\n📋 Checking {len(expected_agents)} expected agents:")
    print("-" * 40)
    
    available_agents = []
    missing_agents = []
    
    for agent_name, agent_type in expected_agents.items():
        try:
            # Try to create agent
            agent = factory.create_agent(agent_type)
            if agent:
                available_agents.append(agent_name)
                print(f"✅ {agent_name:<20} -> {agent_type.value:<25} [Available]")
            else:
                missing_agents.append(agent_name)
                print(f"❌ {agent_name:<20} -> {agent_type.value:<25} [Not Found]")
        except Exception as e:
            missing_agents.append(agent_name)
            print(f"❌ {agent_name:<20} -> {agent_type.value:<25} [Error: {str(e)[:30]}...]")
    
    # Check what's actually registered
    print(f"\n🏭 AgentFactory Registry Contents:")
    print("-" * 40)
    
    if hasattr(factory, '_registry') and hasattr(factory._registry, '_agents'):
        registered_types = list(factory._registry._agents.keys())
        print(f"📊 Registered agent types ({len(registered_types)}):")
        for agent_type in registered_types:
            agent_class = factory._registry._agents[agent_type].__name__ if factory._registry._agents[agent_type] else "Unknown"
            print(f"   • {agent_type.value:<25} -> {agent_class}")
    else:
        print("ℹ️  Could not access registry contents (might be private)")
    
    # Check for workflow support
    print(f"\n🔄 Workflow Support Check:")
    print("-" * 40)
    
    if hasattr(factory, '_registry') and hasattr(factory._registry, '_workflows'):
        workflow_count = len(factory._registry._workflows)
        print(f"📈 LangGraph workflows registered: {workflow_count}")
        for agent_type in factory._registry._workflows:
            print(f"   • {agent_type.value} has LangGraph workflow")
    else:
        print("ℹ️  LangGraph workflow support not detected")
    
    # Summary
    print(f"\n📈 Summary:")
    print("=" * 60)
    print(f"✅ Available agents:    {len(available_agents)}/{len(expected_agents)}")
    print(f"❌ Missing agents:      {len(missing_agents)}/{len(expected_agents)}")
    
    if available_agents:
        print(f"\n✅ Ready for execution: {', '.join(available_agents)}")
    
    if missing_agents:
        print(f"\n❌ Need registration:   {', '.join(missing_agents)}")
        print("\n💡 To fix missing agents:")
        print("   1. Check agent imports in src/main.py")
        print("   2. Ensure agent classes are properly registered")
        print("   3. Verify agent constructor signatures")
    
    # Test basic workflow sequence
    print(f"\n🧪 Testing Basic Workflow Sequence:")
    print("-" * 40)
    
    basic_sequence = ["planner", "researcher", "writer", "editor"]
    workflow_ready = True
    
    for agent_name in basic_sequence:
        if agent_name in available_agents:
            print(f"✅ {agent_name} ready for workflow")
        else:
            print(f"❌ {agent_name} NOT ready - workflow will fail")
            workflow_ready = False
    
    if workflow_ready:
        print("\n🎉 Basic content generation workflow is ready!")
        print("   You can execute: Planner → Researcher → Writer → Editor")
    else:
        print("\n⚠️  Basic workflow NOT ready - missing required agents")
    
    print(f"\n🚀 Workflow Executor Status: {'READY' if workflow_ready else 'NEEDS SETUP'}")
    
    return len(missing_agents) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)