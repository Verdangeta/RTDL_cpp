#ifndef RTDLITE_LINK_CUT_TREE_H
#define RTDLITE_LINK_CUT_TREE_H

#include <algorithm>
#include <vector>

class LinkCutTree {
  public:
    static const int NEG_KEY = -1;

    struct PathMax {
        int key;
        int node;
    };

    explicit LinkCutTree(int node_count = 0, int reserve_count = 0) : nodes(node_count) {
        if (reserve_count > node_count) {
            nodes.reserve(reserve_count);
        }
        for (int i = 0; i < node_count; ++i) {
            nodes[i].max_node = i;
        }
    }

    int add_node(int key) {
        Node node;
        node.key = key;
        node.max_key = key;
        node.max_node = (int)nodes.size();
        nodes.push_back(node);
        return (int)nodes.size() - 1;
    }

    void make_root(int x) {
        access(x);
        nodes[x].rev = !nodes[x].rev;
    }

    void link(int u, int v) {
        make_root(u);
        nodes[u].parent = v;
    }

    void cut(int u, int v) {
        make_root(u);
        access(v);
        if (nodes[v].child[0] == u && nodes[u].child[1] == -1) {
            nodes[v].child[0] = -1;
            nodes[u].parent = -1;
            pull(v);
        }
    }

    PathMax query_path_max(int u, int v) {
        make_root(u);
        access(v);
        return {nodes[v].max_key, nodes[v].max_node};
    }

  private:
    struct Node {
        int child[2] = {-1, -1};
        int parent = -1;
        bool rev = false;
        int key = NEG_KEY;
        int max_key = NEG_KEY;
        int max_node = -1;
    };

    std::vector<Node> nodes;

    bool is_splay_root(int x) const {
        int p = nodes[x].parent;
        return p == -1 || (nodes[p].child[0] != x && nodes[p].child[1] != x);
    }

    void push(int x) {
        if (x == -1 || !nodes[x].rev) {
            return;
        }
        std::swap(nodes[x].child[0], nodes[x].child[1]);
        if (nodes[x].child[0] != -1) {
            nodes[nodes[x].child[0]].rev = !nodes[nodes[x].child[0]].rev;
        }
        if (nodes[x].child[1] != -1) {
            nodes[nodes[x].child[1]].rev = !nodes[nodes[x].child[1]].rev;
        }
        nodes[x].rev = false;
    }

    void pull(int x) {
        nodes[x].max_key = nodes[x].key;
        nodes[x].max_node = x;
        for (int side = 0; side < 2; ++side) {
            int child = nodes[x].child[side];
            if (child != -1 && nodes[child].max_key > nodes[x].max_key) {
                nodes[x].max_key = nodes[child].max_key;
                nodes[x].max_node = nodes[child].max_node;
            }
        }
    }

    void push_path(int x) {
        if (!is_splay_root(x)) {
            push_path(nodes[x].parent);
        }
        push(x);
    }

    void rotate(int x) {
        int p = nodes[x].parent;
        int g = nodes[p].parent;
        push(p);
        push(x);
        int is_right = (nodes[p].child[1] == x);
        int b = nodes[x].child[is_right ^ 1];

        if (!is_splay_root(p)) {
            if (nodes[g].child[0] == p) {
                nodes[g].child[0] = x;
            } else {
                nodes[g].child[1] = x;
            }
        }
        nodes[x].parent = g;

        nodes[x].child[is_right ^ 1] = p;
        nodes[p].parent = x;
        nodes[p].child[is_right] = b;
        if (b != -1) {
            nodes[b].parent = p;
        }

        pull(p);
        pull(x);
    }

    void splay(int x) {
        push_path(x);
        while (!is_splay_root(x)) {
            int p = nodes[x].parent;
            int g = nodes[p].parent;
            if (!is_splay_root(p)) {
                bool zigzig = (nodes[p].child[0] == x) == (nodes[g].child[0] == p);
                rotate(zigzig ? p : x);
            }
            rotate(x);
        }
    }

    void access(int x) {
        int last = -1;
        for (int y = x; y != -1; y = nodes[y].parent) {
            splay(y);
            nodes[y].child[1] = last;
            if (last != -1) {
                nodes[last].parent = y;
            }
            pull(y);
            last = y;
        }
        splay(x);
    }
};

#endif /* RTDLITE_LINK_CUT_TREE_H */
