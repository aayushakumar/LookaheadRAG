"""
Tests for Variable Binding functionality.

Tests the new binding-related features:
- ProducedVariable and EntityType
- PlanNode binding fields (produces, bindings, required_inputs)
- Binding validation and resolution
- BindingResolver extraction
"""

import pytest
from src.planner.schema import (
    PlanNode,
    PlanGraph,
    OperatorType,
    EntityType,
    ProducedVariable,
)
from src.planner.binding_resolver import (
    BindingResolver,
    BindingContext,
    ExtractedEntity,
)


class TestProducedVariable:
    """Tests for ProducedVariable schema."""
    
    def test_create_basic(self):
        """Test creating a basic ProducedVariable."""
        pv = ProducedVariable(
            var="person",
            type=EntityType.PERSON,
            description="Oscar-winning actress",
        )
        assert pv.var == "person"
        assert pv.type == EntityType.PERSON
        assert pv.description == "Oscar-winning actress"
    
    def test_default_type(self):
        """Test default entity type."""
        pv = ProducedVariable(var="entity")
        assert pv.type == EntityType.OTHER
    
    def test_to_dict(self):
        """Test serialization."""
        pv = ProducedVariable(
            var="director",
            type=EntityType.PERSON,
            description="Film director",
        )
        d = pv.to_dict()
        assert d["var"] == "director"
        assert d["type"] == "person"
        assert d["description"] == "Film director"


class TestPlanNodeBindings:
    """Tests for PlanNode binding functionality."""
    
    def test_node_with_produces(self):
        """Test creating a node with produces definitions."""
        node = PlanNode(
            id="n1",
            query="La La Land Oscar winner actress",
            op=OperatorType.LOOKUP,
            confidence=0.85,
            produces=[
                ProducedVariable(
                    var="person",
                    type=EntityType.PERSON,
                    description="Oscar-winning actress",
                )
            ],
        )
        assert len(node.produces) == 1
        assert node.produces[0].var == "person"
    
    def test_node_with_bindings(self):
        """Test creating a node with bindings and required_inputs."""
        node = PlanNode(
            id="n2",
            query="{actress} filmography movies",
            op=OperatorType.BRIDGE,
            depends_on=["n1"],
            confidence=0.75,
            bindings={"actress": "n1.person"},
            required_inputs=["actress"],
        )
        assert node.bindings["actress"] == "n1.person"
        assert "actress" in node.required_inputs
    
    def test_required_input_without_binding_raises(self):
        """Test that required_inputs must have bindings."""
        with pytest.raises(ValueError, match="no corresponding binding"):
            PlanNode(
                id="n2",
                query="{actress} films",
                op=OperatorType.BRIDGE,
                required_inputs=["actress"],  # No binding for this
                bindings={},
            )
    
    def test_has_placeholders(self):
        """Test placeholder detection."""
        node_with = PlanNode(
            id="n1",
            query="{person} filmography",
        )
        node_without = PlanNode(
            id="n2",
            query="Emma Stone filmography",
        )
        assert node_with.has_placeholders() is True
        assert node_without.has_placeholders() is False
    
    def test_can_execute(self):
        """Test execution readiness check."""
        node = PlanNode(
            id="n2",
            query="{actress} films",
            bindings={"actress": "n1.person"},
            required_inputs=["actress"],
        )
        
        # Cannot execute without resolved vars
        assert node.can_execute(set()) is False
        assert node.can_execute({"other_var"}) is False
        
        # Can execute with required var
        assert node.can_execute({"actress"}) is True
    
    def test_resolve_bindings(self):
        """Test binding resolution."""
        node = PlanNode(
            id="n2",
            query="{actress} filmography movies",
            bindings={"actress": "n1.person"},
            required_inputs=["actress"],
        )
        
        context = {"n1.person": "Emma Stone"}
        resolved = node.resolve_bindings(context)
        
        assert resolved == "Emma Stone filmography movies"
        assert node.bound_query == "Emma Stone filmography movies"
        assert "actress" in node.binding_citations
    
    def test_get_effective_query(self):
        """Test getting query for retrieval."""
        node = PlanNode(
            id="n1",
            query="{person} movies",
        )
        
        # Before resolution, returns original
        assert node.get_effective_query() == "{person} movies"
        
        # After resolution, returns bound
        node.bound_query = "Emma Stone movies"
        assert node.get_effective_query() == "Emma Stone movies"
    
    def test_to_dict_includes_bindings(self):
        """Test that to_dict includes binding fields."""
        node = PlanNode(
            id="n2",
            query="{person} films",
            produces=[
                ProducedVariable(var="film", type=EntityType.WORK_OF_ART)
            ],
            bindings={"person": "n1.person"},
            required_inputs=["person"],
        )
        
        d = node.to_dict()
        assert "produces" in d
        assert "bindings" in d
        assert "required_inputs" in d
        assert d["bindings"]["person"] == "n1.person"


class TestBindingContext:
    """Tests for BindingContext."""
    
    def test_empty_context(self):
        """Test empty binding context."""
        ctx = BindingContext()
        assert ctx.get_value("n1.person") is None
        assert ctx.to_context_dict() == {}
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving from context."""
        ctx = BindingContext()
        ctx.resolved["n1.person"] = ExtractedEntity(
            value="Emma Stone",
            entity_type=EntityType.PERSON,
            citation="[n1.2]",
        )
        
        assert ctx.get_value("n1.person") == "Emma Stone"
        assert ctx.get_citation("n1.person") == "[n1.2]"
    
    def test_to_context_dict(self):
        """Test converting to simple dict."""
        ctx = BindingContext()
        ctx.resolved["n1.person"] = ExtractedEntity(
            value="Emma Stone",
            entity_type=EntityType.PERSON,
            citation="[n1.2]",
        )
        ctx.resolved["n1.year"] = ExtractedEntity(
            value="2016",
            entity_type=EntityType.DATE,
            citation="[n1.1]",
        )
        
        d = ctx.to_context_dict()
        assert d["n1.person"] == "Emma Stone"
        assert d["n1.year"] == "2016"


class TestBindingResolver:
    """Tests for BindingResolver."""
    
    def test_resolve_node_no_inputs(self):
        """Test resolving a node with no required inputs."""
        resolver = BindingResolver()
        node = PlanNode(
            id="n1",
            query="La La Land Oscar winner",
        )
        ctx = BindingContext()
        
        result = resolver.resolve_node(node, ctx)
        assert result is True  # No inputs needed
    
    def test_resolve_node_with_context(self):
        """Test resolving a node with available context."""
        resolver = BindingResolver()
        node = PlanNode(
            id="n2",
            query="{actress} filmography",
            bindings={"actress": "n1.person"},
            required_inputs=["actress"],
        )
        
        ctx = BindingContext()
        ctx.resolved["n1.person"] = ExtractedEntity(
            value="Emma Stone",
            entity_type=EntityType.PERSON,
            citation="[n1.1]",
        )
        
        result = resolver.resolve_node(node, ctx)
        assert result is True
        assert node.bound_query == "Emma Stone filmography"
    
    def test_resolve_node_missing_context(self):
        """Test resolving a node with missing context."""
        resolver = BindingResolver()
        node = PlanNode(
            id="n2",
            query="{actress} filmography",
            bindings={"actress": "n1.person"},
            required_inputs=["actress"],
        )
        
        ctx = BindingContext()  # Empty - no n1.person
        
        result = resolver.resolve_node(node, ctx)
        assert result is False  # Should fail


class TestPlanGraphWithBindings:
    """Tests for PlanGraph with binding-enabled nodes."""
    
    def test_create_binding_graph(self):
        """Test creating a PlanGraph with bindings."""
        plan = PlanGraph(
            question="Who directed films starring the La La Land Oscar winner?",
            nodes=[
                PlanNode(
                    id="n1",
                    query="La La Land Oscar winner actress",
                    op=OperatorType.LOOKUP,
                    confidence=0.85,
                    produces=[
                        ProducedVariable(
                            var="person",
                            type=EntityType.PERSON,
                            description="Oscar-winning actress",
                        )
                    ],
                ),
                PlanNode(
                    id="n2",
                    query="{person} filmography movies director",
                    op=OperatorType.BRIDGE,
                    depends_on=["n1"],
                    confidence=0.75,
                    bindings={"person": "n1.person"},
                    required_inputs=["person"],
                ),
            ],
        )
        
        assert len(plan.nodes) == 2
        assert plan.nodes[1].required_inputs == ["person"]
        
        # Verify execution order respects dependencies
        order = plan.get_execution_order()
        assert len(order) == 2
        assert order[0][0].id == "n1"
        assert order[1][0].id == "n2"
