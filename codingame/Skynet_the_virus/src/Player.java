import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

enum Color
{
    green, yellow, red
}

class Edge
{
    public Vertex node1;
    public Vertex node2;

    public Edge(Vertex node1_, Vertex node2_)
    {
        node1 = node1_;
        node2 = node2_;
    }

    public Vertex get_opposite_end(Vertex vertex)
    {
        return node1.id == vertex.id ? node2 : node1;
    }
}

class Vertex
{
    public int id;
    public ArrayList<Edge> edges = new ArrayList<>();
    public Color color = Color.green;
    public int red_neighbours = 0;
    public int component = -1;
    public boolean is_visited = false;

    public Vertex(int id_)
    {
        id = id_;
    }

    public void make_red()
    {
        color = Color.red;
        for (Edge edge: edges)
        {
            Vertex opposite = edge.get_opposite_end(this);
            opposite.red_neighbours += 1;
        }
    }
    public void make_yellow()
    {
        color = Color.yellow;
    }

    public void make_green()
    {
        color = Color.green;
    }
}

class Graph
{
    public ArrayList<Vertex> vertices;

    public void add_link(Vertex node1, Vertex node2)
    {
        Edge new_edge = new Edge(node1, node2);
        node1.edges.add(new_edge);
        node2.edges.add(new_edge);
    }

    public static Graph initialize_from_stream(Scanner in)
    {
        Graph new_graph = new Graph();

        int nodes_amount = in.nextInt(); // the total number of nodes in the level, including the gateways
        new_graph.vertices = new ArrayList<>(IntStream.range(0, nodes_amount + 1).mapToObj(Vertex::new).collect(Collectors.toList()));

        int links_amount = in.nextInt(); // the number of links
        int exits_amount = in.nextInt(); // the number of exit gateways

        System.err.println(nodes_amount);
        System.err.println(links_amount);
        System.err.println(exits_amount);

        for (int i = 0; i < links_amount; ++i)
        {
            int node1_id = in.nextInt(); // N1 and N2 defines a link between these nodes
            int node2_id = in.nextInt();
            new_graph.add_link(new_graph.vertices.get(node1_id), new_graph.vertices.get(node2_id));

            System.err.format("%d %d\n", node1_id, node2_id);
        }

        for (int i = 0; i < exits_amount; ++i)
        {
            int exit_index = in.nextInt(); // the index of a gateway node
            new_graph.add_link(new_graph.vertices.get(exit_index), new_graph.vertices.get(new_graph.vertices.size() - 1));

            System.err.format("%d\n", exit_index);
        }

        return new_graph;
    }

    public int[][] get_distance_matrix()
    {
        int[][] distance_matrix = new int[vertices.size()][vertices.size()];
        for (int i = 0; i < vertices.size(); ++i)
        {
            for (int j = 0; j < vertices.size(); ++j)
            {
                if (i == j)
                {
                    distance_matrix[i][j] = 0;
                }
                else
                {
                    distance_matrix[i][j] = 1000000;
                }
            }
        }

        for (Vertex vertex: vertices)
        {
            for (Edge edge: vertex.edges)
            {
                Vertex opposite = edge.get_opposite_end(vertex);
                distance_matrix[vertex.id][opposite.id] = 1;
            }
        }

        for (int k = 0; k < vertices.size(); ++k)
        {
            for (int i = 0; i < vertices.size(); ++i)
            {
                for (int j = 0; j < vertices.size(); ++j)
                {
                    distance_matrix[i][j] = Integer.min(distance_matrix[i][j], distance_matrix[i][k] + distance_matrix[k][j]);
                }
            }
        }
        return distance_matrix;
    }
}

class Player
{
    public static List<Vertex> get_next_layer(List<Vertex> current_layer)
    {
        List<Vertex> result = new ArrayList<>();
        for (Vertex vertex: current_layer)
        {
            vertex.make_red();
            for (Edge edge: vertex.edges)
            {
                Vertex opposite = edge.get_opposite_end(vertex);
                if (opposite.color == Color.green)
                {
                    opposite.make_yellow();
                    result.add(opposite);
                }
            }
        }
        return result;
    }

    public static int find_cut_edge_index(Vertex border_vertex)
    {
        for (int i = 0; i < border_vertex.edges.size(); ++i)
        {
            Edge edge = border_vertex.edges.get(i);
            Vertex opposite = edge.get_opposite_end(border_vertex);
            if (opposite.color == Color.red)
            {
                return i;
            }
        }
        return -1;
    }

    public static void report_cut_edge(Vertex border_vertex)
    {
        int cut_edge_index = find_cut_edge_index(border_vertex);
        Edge cut_edge = border_vertex.edges.get(cut_edge_index);
        border_vertex.edges.remove(cut_edge_index);
        border_vertex.red_neighbours -= 1;
        System.out.format("%d %d\n", cut_edge.node1.id, cut_edge.node2.id);
    }

    public static void assign_component(Vertex start_vertex, int component, List<Vertex> component_nodes)
    {
        start_vertex.is_visited = true;
        start_vertex.component = component;
        component_nodes.add(start_vertex);
        for (Edge edge: start_vertex.edges)
        {
            Vertex opposite = edge.get_opposite_end(start_vertex);
            if (opposite.color == Color.yellow && !opposite.is_visited)
            {
                assign_component(opposite, component, component_nodes);
            }
        }
    }

    public static Map<Integer, List<Vertex>> compute_components(List<Vertex> current_layer)
    {
        Map<Integer, List<Vertex>> result = new HashMap<>();
        int last_component = 0;
        for (Vertex vertex: current_layer)
        {
            if (!vertex.is_visited)
            {
                List<Vertex> component_nodes = new ArrayList<>();
                assign_component(vertex, last_component, component_nodes);
                result.put(last_component, component_nodes);
                last_component += 1;
            }
        }
        return result;
    }

    public static Map<Integer, List<Vertex>> get_critical_nodes(Map<Integer, List<Vertex>> component_nodes)
    {
        Map<Integer, List<Vertex>> critical_nodes = new HashMap<>();
        for (Map.Entry<Integer, List<Vertex>> entry: component_nodes.entrySet())
        {
            critical_nodes.put(entry.getKey(), entry.getValue().stream().filter(v -> v.red_neighbours > 1).collect(Collectors.toList()));
        }
        return critical_nodes;
    }

    public static void filter_components(Map<Integer, List<Vertex>> component_nodes, Map<Integer, List<Vertex>> component_critical_nodes)
    {
        List<Integer> key_to_remove = component_nodes.keySet().stream().filter(key -> component_critical_nodes.get(key).size() == 0).
                collect(Collectors.toList());
        for (int key: key_to_remove)
        {
            component_nodes.remove(key);
            component_critical_nodes.remove(key);
        }
    }

    public static void main(String args[])
    {
        try
        {
            System.setIn(new FileInputStream("input"));
        }
        catch (Exception e)
        {

        }

        Scanner in = new Scanner(System.in);
        Graph game_graph = Graph.initialize_from_stream(in);
        Vertex common_exit = game_graph.vertices.get(game_graph.vertices.size() - 1);
        common_exit.color = Color.yellow;
        List<Vertex> exits = get_next_layer(Collections.singletonList(common_exit));
        List<Vertex> current_layer = get_next_layer(exits);
        int[][] distance_matrix = game_graph.get_distance_matrix();

        Map<Integer, List<Vertex>> component_nodes = compute_components(current_layer);
        Map<Integer, List<Vertex>> critical_nodes = get_critical_nodes(component_nodes);
        filter_components(component_nodes, critical_nodes);
        int total_critical_size = critical_nodes.values().stream().mapToInt(List::size).sum();

        int skynet_index = in.nextInt();
        int last_index = 0;
        while (true)
        {
            Vertex skynet_vertex = game_graph.vertices.get(skynet_index);
            if (skynet_vertex.color == Color.yellow)
            {
                report_cut_edge(skynet_vertex);
                skynet_vertex.make_green();
            }
            else if (total_critical_size > 0)
            {
                final int skynet_index_copy = skynet_index;
                Vertex closest_vertex = component_nodes.values().stream().flatMap(Collection::stream).
                        min((v1, v2) -> distance_matrix[v1.id][skynet_index_copy] - distance_matrix[v2.id][skynet_index_copy]).get();
                Vertex closest_critical_vertex = critical_nodes.get(closest_vertex.component).get(0);
                critical_nodes.get(closest_vertex.component).remove(0);
                if (critical_nodes.get(closest_vertex.component).size() == 0)
                {
                    critical_nodes.remove(closest_vertex.component);
                    component_nodes.remove(closest_vertex.component);
                }
                report_cut_edge(closest_critical_vertex);
                total_critical_size -= 1;
            }
            else
            {
                for (; last_index < current_layer.size(); ++last_index)
                {
                    Vertex current = current_layer.get(last_index);
                    if (current.color == Color.yellow)
                    {
                        report_cut_edge(current);
                        current.make_green();
                        break;
                    }
                }
            }
            skynet_index = in.nextInt();

            System.err.println(skynet_index);
        }
    }
}