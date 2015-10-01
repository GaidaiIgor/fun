import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;

class Bucket
{
    Bucket adjacent = null;
    int capacity = 0;
    Node parent = null;
    int edges = 0;

    public Bucket(Node parent_)
    {
        parent = parent_;
    }
    public Bucket() {}
}

class Node
{
    int id;
    int links_left;
    int x;
    int y;
    List<Bucket> buckets = new ArrayList<>(4);

    static int last_id = 0;

    public Node(int id_, int links_left_, int x_, int y_)
    {
        id = id_;
        links_left = links_left_;
        x = x_;
        y = y_;
        for (int i = 0; i < 4; ++i)
        {
            buckets.add(new Bucket(this));
        }
    }
}

class Player
{
    List<List<Integer>> map;
    final int max_capacity = 2;

    public List<Node> read_map()
    {
        List<Node> all_nodes = new ArrayList<>();
        Scanner in = new Scanner(System.in);
        int width = in.nextInt(); // the number of cells on the X axis
        System.err.println(width);
        int height = in.nextInt(); // the number of cells on the Y axis
        System.err.println(height);
        in.nextLine();
        map = new ArrayList<>(height);
        for (int i = 0; i < height; ++i)
        {
            String line = in.nextLine(); // width characters, each either a number or a '.'
            System.err.println(line);
            map.add(new ArrayList<>(width));
            for (int j = 0; j < width; ++j)
            {
                if (line.charAt(j) == '.')
                {
                    map.get(i).add(-1);
                }
                else
                {
                    int degree = Integer.parseInt("" + line.charAt(j));
                    Node new_node = new Node(Node.last_id++, degree, j, i);
                    map.get(i).add(new_node.id);
                    all_nodes.add(new_node);
                }
            }
        }
        return all_nodes;
    }

    public void set_relationship(Bucket bucket1, Bucket bucket2)
    {
        bucket1.adjacent = bucket2;
        bucket2.adjacent = bucket1;
        int capacity = Collections.min(Arrays.asList(bucket1.parent.links_left, bucket2.parent.links_left, max_capacity));
        bucket1.capacity = capacity;
        bucket2.capacity = capacity;
    }

    public int find_up_neighbour(int x, int y)
    {
        for (int i = y - 1; i >= 0; --i)
        {
            int next_id = map.get(i).get(x);
            if (next_id != -1)
            {
                return next_id;
            }
        }
        return -1;
    }

    public int find_right_neighbour(int x, int y)
    {
        for (int i = x + 1; i < map.get(0).size(); ++i)
        {
            int next_id = map.get(y).get(i);
            if (next_id != -1)
            {
                return next_id;
            }
        }
        return -1;
    }

    public int find_down_neighbour(int x, int y)
    {
        for (int i = y + 1; i < map.size(); ++i)
        {
            int next_id = map.get(i).get(x);
            if (next_id != -1)
            {
                return next_id;
            }
        }
        return -1;
    }

    public int find_left_neighbour(int x, int y)
    {
        for (int i = x - 1; i >= 0; --i)
        {
            int next_id = map.get(y).get(i);
            if (next_id != -1)
            {
                return next_id;
            }
        }
        return -1;
    }

    public void acquaint_nodes(List<Node> nodes)
    {
        for (List<Integer> row: map)
        {
            row.stream().filter(next_id -> next_id != -1).forEach(next_id -> {
                Node node = nodes.get(next_id);
                int right_neighbour_id = find_right_neighbour(node.x, node.y);
                Node right_neighbour = right_neighbour_id == -1 ? null : nodes.get(right_neighbour_id);
                if (right_neighbour != null)
                {
                    set_relationship(node.buckets.get(1), right_neighbour.buckets.get(3));
                }

                int down_neighbour_id = find_down_neighbour(node.x, node.y);
                Node down_neighbour = down_neighbour_id == -1 ? null : nodes.get(down_neighbour_id);
                if (down_neighbour != null)
                {
                    set_relationship(node.buckets.get(2), down_neighbour.buckets.get(0));
                }
            });
        }
    }

    public void draw_horizontal_edge(int x1_init, int x2_init, int y, List<Node> nodes)
    {
        int x1 = Integer.min(x1_init, x2_init) + 1;
        int x2 = Integer.max(x1_init, x2_init);
        for (int i = x1; i < x2; ++i)
        {
            int up_neighbour_id = find_up_neighbour(i, y);
            Node up_neighbour = up_neighbour_id == -1 ? null : nodes.get(up_neighbour_id);
            if (up_neighbour != null)
            {
                up_neighbour.buckets.get(2).capacity = 0;
            }

            int down_neighbour_id = find_down_neighbour(i, y);
            Node down_neighbour = down_neighbour_id == -1 ? null : nodes.get(down_neighbour_id);
            if (down_neighbour != null)
            {
                down_neighbour.buckets.get(0).capacity = 0;
            }
        }
    }

    public void draw_vertical_edge(int y1_init, int y2_init, int x, List<Node> nodes)
    {
        int y1 = Integer.min(y1_init, y2_init) + 1;
        int y2 = Integer.max(y1_init, y2_init);
        for (int i = y1; i < y2; ++i)
        {
            int left_neighbour_id = find_left_neighbour(x, i);
            Node left_neighbour = left_neighbour_id == -1 ? null : nodes.get(left_neighbour_id);
            if (left_neighbour != null)
            {
                left_neighbour.buckets.get(1).capacity = 0;
            }

            int right_neighbour_id = find_right_neighbour(x, i);
            Node right_neighbour = right_neighbour_id == -1 ? null : nodes.get(right_neighbour_id);
            if (right_neighbour != null)
            {
                right_neighbour.buckets.get(3).capacity = 0;
            }
        }
    }

    public void balance_capacity(Node node)
    {
        for (Bucket gate: node.buckets)
        {
            gate.capacity = Integer.min(gate.capacity, node.links_left);
            if (gate.adjacent != null)
            {
                gate.adjacent.capacity = Integer.min(gate.capacity, gate.adjacent.capacity);
            }
        }
    }

    public void establish_link(Bucket bucket)
    {
        bucket.capacity -= 1;
        bucket.parent.links_left -= 1;
        bucket.edges += 1;
    }

    public void add_edge(Bucket bucket, List<Node> board_state)
    {
        if (bucket.edges == 0)
        {
            if (bucket.parent.y == bucket.adjacent.parent.y)
            {
                draw_horizontal_edge(bucket.parent.x, bucket.adjacent.parent.x, bucket.parent.y, board_state);
            }
            else
            {
                draw_vertical_edge(bucket.parent.y, bucket.adjacent.parent.y, bucket.parent.x, board_state);
            }
        }

        establish_link(bucket);
        establish_link(bucket.adjacent);
        balance_capacity(bucket.parent);
        balance_capacity(bucket.adjacent.parent);
    }

    public boolean make_mandatory_links(List<Node> board_state)
    {
        int mandatory_edges = 0;
        for (Node node: board_state)
        {
            if (node.links_left == 0)
            {
                continue;
            }
            Integer capacity_sum = node.buckets.stream().reduce(0, (acc, v) -> acc + v.capacity, (a, b) -> a + b);
            if (node.links_left > capacity_sum)
            {
                return false;
            }
            for (int capacity = max_capacity; capacity > 0; --capacity)
            {
                final int current_capacity = capacity;
                List<Bucket> current_buckets = node.buckets.stream().filter(b -> b.capacity == current_capacity).
                        collect(Collectors.toList());
                if (current_buckets.size() > 0 && node.links_left > capacity_sum - capacity)
                {
                    current_buckets.forEach(b -> add_edge(b, board_state));
                    mandatory_edges += current_buckets.size();
                    capacity_sum -= current_buckets.size();
                }
            }
        }
        return mandatory_edges == 0 || make_mandatory_links(board_state);
    }

    public boolean no_links_left(List<Node> board_state)
    {
        return board_state.stream().allMatch(n -> n.links_left == 0);
    }

    public void dfs(Node start, List<Boolean> visited)
    {
        visited.set(start.id, true);
        start.buckets.stream().filter(bucket -> bucket.edges > 0 && !visited.get(bucket.adjacent.parent.id)).
                forEach(bucket -> dfs(bucket.adjacent.parent, visited));
    }

    public boolean is_connected(List<Node> board_state)
    {
        List<Boolean> visited = new ArrayList<>(Collections.nCopies(board_state.size(), false));
        dfs(board_state.get(0), visited);
        return visited.stream().allMatch(b -> b);
    }

    public Bucket copy_bucket(Bucket other, Map<Bucket, Bucket> buckets_images, Map<Node, Node> nodes_images)
    {
        Bucket copy = buckets_images.get(other);
        if (copy == null)
        {
            copy = new Bucket();
            buckets_images.put(other, copy);
            copy.parent = copy_node(other.parent, buckets_images, nodes_images);
            copy.capacity = other.capacity;
            copy.edges = other.edges;
            copy.adjacent = other.adjacent == null ? null : copy_bucket(other.adjacent, buckets_images, nodes_images);
        }
        return copy;
    }

    public Node copy_node(Node other, Map<Bucket, Bucket> buckets_images, Map<Node, Node> nodes_images)
    {
        Node copy = nodes_images.get(other);
        if (copy == null)
        {
            copy = new Node(other.id, other.links_left, other.x, other.y);
            nodes_images.put(other, copy);
            for (int i = 0; i < other.buckets.size(); ++i)
            {
                Bucket original = other.buckets.get(i);
                Bucket bucket_image = copy_bucket(original, buckets_images, nodes_images);
                copy.buckets.set(i, bucket_image);
            }
        }
        return copy;
    }

    public List<Node> copy_state(List<Node> board_state, Map<Bucket, Bucket> buckets_images, Map<Node, Node> nodes_images)
    {
        List<Node> copy = new ArrayList<>(board_state.size());
        copy.addAll(board_state.stream().map(node -> copy_node(node, buckets_images, nodes_images)).collect(Collectors.toList()));
        return copy;
    }

    public List<Node> make_links(List<Node> board_state)
    {
        if (!make_mandatory_links(board_state))
        {
            return null;
        }
        if (no_links_left(board_state))
        {
            return is_connected(board_state) ? board_state : null;
        }
        Node unresolved = board_state.stream().filter(n -> n.links_left > 0).findFirst().get();
        for (Bucket bucket: unresolved.buckets)
        {
            if (bucket.capacity > 0)
            {
                Map<Node, Node> nodes_images = new IdentityHashMap<>();
                Map<Bucket, Bucket> buckets_images = new IdentityHashMap<>();
                List<Node> board_state_copy = copy_state(board_state, buckets_images, nodes_images);
                Bucket bucket_image = buckets_images.get(bucket);
                add_edge(bucket_image, board_state_copy);
                List<Node> result = make_links(board_state_copy);
                if (result != null)
                {
                    return result;
                }
            }
        }
        return null;
    }

    public void report_edges(List<Node> resolved)
    {
        for (Node node: resolved)
        {
            for (int i = 1; i <= 2; ++i)
            {
                Bucket bucket = node.buckets.get(i);
                if (bucket.edges > 0)
                {
                    System.out.format("%d %d %d %d %d\n", node.x, node.y, bucket.adjacent.parent.x, bucket.adjacent.parent.y,
                            bucket.edges);
                }
            }
        }
    }

    public void play()
    {
        List<Node> nodes = read_map();
        acquaint_nodes(nodes);
        List<Node> resolved = make_links(nodes);
        report_edges(resolved);
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

        Player player = new Player();
        player.play();
    }
}