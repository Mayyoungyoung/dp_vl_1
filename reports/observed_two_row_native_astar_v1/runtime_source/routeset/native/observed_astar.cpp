// Exact graph/search order adapter for scripts/observation_multiroute_astar_v2.py.
// No geometry construction, path repair, label access, or candidate resampling.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

namespace {
using Clock = std::chrono::steady_clock;
struct Item { double f; uint64_t order; int32_t node; };
struct Later {
    bool operator()(const Item& a, const Item& b) const {
        if (a.f != b.f) return a.f > b.f;
        return a.order > b.order;
    }
};
uint64_t edge_key(int32_t a, int32_t b) {
    if (b < a) std::swap(a, b);
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}
}

extern "C" EXPORT int observed_astar_abi_version() { return 1; }

extern "C" EXPORT int observed_astar_v1(
    int32_t nx, int32_t ny, int32_t nz,
    const uint8_t* free_cells, const uint8_t* start_permission, const uint8_t* target_permission,
    const double* lower, const double* exact_goal,
    int32_t start_count, const int32_t* start_ids, const double* start_lengths, const double* start_penalties,
    int32_t goal_count, const int32_t* goal_ids, const double* goal_lengths, const double* goal_penalties,
    const int32_t* deltas, const int32_t* touch_prefix, const int32_t* touches, const double* lengths,
    int32_t edge_count, const int32_t* edge_ids, const double* edge_penalties, const double* field,
    int32_t maximum_nodes, double deadline_seconds, double elapsed_before_kernel,
    int32_t* output_path, int32_t output_capacity, int64_t* stats, double* kernel_seconds) {
    const auto began = Clock::now();
    std::fill(stats, stats+14, int64_t(0));
    stats[4] = start_count; stats[5] = goal_count;
    auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now()-began).count(); };
    auto finish = [&](int status) { *kernel_seconds = elapsed(); return status; };
    try {
        const int64_t size64 = int64_t(nx)*ny*nz;
        if (nx <= 0 || ny <= 0 || nz <= 0 || size64 > 2000000 || maximum_nodes <= 0 ||
            output_capacity < maximum_nodes+1 || elapsed_before_kernel < 0.) return finish(-1);
        if (start_count == 0) return finish(1);
        if (goal_count == 0) return finish(2);
        const int32_t size = int32_t(size64), goal_virtual = size, start_virtual = -1;
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> scores(size+1, infinity), goal_cost(size, infinity);
        std::vector<int32_t> previous(size+1, -2);
        std::vector<uint8_t> closed(size+1, 0);
        std::unordered_map<uint64_t, double> edge_costs;
        edge_costs.reserve(edge_count);
        for (int32_t i=0; i<edge_count; ++i)
            edge_costs.emplace(edge_key(edge_ids[2*i], edge_ids[2*i+1]), edge_penalties[i]);
        for (int32_t i=0; i<goal_count; ++i)
            goal_cost[goal_ids[i]] = goal_lengths[i]*(1.+4.*goal_penalties[i]);
        auto coordinates = [&](int32_t id, int32_t* xyz) {
            xyz[0] = id/(ny*nz); xyz[1] = (id/nz)%ny; xyz[2] = id%nz;
        };
        auto heuristic = [&](int32_t id) {
            int32_t xyz[3]; coordinates(id, xyz);
            const double x = (lower[0]+double(xyz[0])*.025)-exact_goal[0];
            const double y = (lower[1]+double(xyz[1])*.025)-exact_goal[1];
            const double z = (lower[2]+double(xyz[2])*.025)-exact_goal[2];
            return std::sqrt((x*x+y*y)+z*z);
        };
        std::priority_queue<Item,std::vector<Item>,Later> queue;
        uint64_t counter = 0;
        for (int32_t i=0; i<start_count; ++i) {
            const int32_t id = start_ids[i];
            const double cost = start_lengths[i]*(1.+4.*start_penalties[i]);
            scores[id] = cost; previous[id] = start_virtual;
            queue.push(Item{cost+heuristic(id), counter++, id});
        }
        while (!queue.empty()) {
            const auto entry = queue.top(); queue.pop();
            const int32_t current = entry.node;
            if (closed[current]) continue;
            if (current == goal_virtual) {
                std::vector<int32_t> path{previous[goal_virtual]};
                while (previous[path.back()] != start_virtual) {
                    if (previous[path.back()] < 0 || int32_t(path.size()) >= output_capacity) return finish(-2);
                    path.push_back(previous[path.back()]);
                }
                std::reverse(path.begin(), path.end());
                std::copy(path.begin(), path.end(), output_path);
                stats[9] = int64_t(path.size())-1; stats[10] = 2; stats[13] = path.size();
                for (size_t i=1; i<path.size(); ++i) {
                    stats[11] += (start_permission[path[i-1]] || start_permission[path[i]]) ? 1 : 0;
                    stats[12] += (target_permission[path[i-1]] || target_permission[path[i]]) ? 1 : 0;
                }
                return finish(0);
            }
            closed[current] = 1;
            ++stats[0];
            if (stats[0] >= maximum_nodes) return finish(3);
            if (stats[0]%64 == 0 && deadline_seconds >= 0. &&
                elapsed_before_kernel+elapsed() >= deadline_seconds) return finish(4);
            if (goal_cost[current] != infinity) {
                ++stats[6];
                const double proposed = scores[current]+goal_cost[current];
                if (proposed < scores[goal_virtual]) {
                    scores[goal_virtual] = proposed; previous[goal_virtual] = current;
                    queue.push(Item{proposed,counter++,goal_virtual});
                }
            }
            int32_t xyz[3]; coordinates(current, xyz);
            for (int32_t n=0; n<26; ++n) {
                ++stats[1];
                const int32_t a = xyz[0]+deltas[3*n], b = xyz[1]+deltas[3*n+1], c = xyz[2]+deltas[3*n+2];
                if (a<0 || b<0 || c<0 || a>=nx || b>=ny || c>=nz) continue;
                bool blocked = false;
                for (int32_t j=touch_prefix[n]; j<touch_prefix[n+1]; ++j) {
                    ++stats[3];
                    const int32_t cell = ((xyz[0]+touches[3*j])*ny+(xyz[1]+touches[3*j+1]))*nz+xyz[2]+touches[3*j+2];
                    if (!free_cells[cell]) { blocked = true; break; }
                }
                if (blocked) continue;
                const int32_t neighbor = (a*ny+b)*nz+c;
                ++stats[2];
                stats[7] += (start_permission[current] || start_permission[neighbor]) ? 1 : 0;
                stats[8] += (target_permission[current] || target_permission[neighbor]) ? 1 : 0;
                double penalty = 0.;
                if (field) penalty = (field[current]+field[neighbor])*.5;
                else {
                    auto found = edge_costs.find(edge_key(current,neighbor));
                    if (found != edge_costs.end()) penalty = found->second;
                }
                const double cost = (lengths[n]*.025)*(1.+4.*penalty);
                const double proposed = scores[current]+cost;
                if (proposed < scores[neighbor]) {
                    scores[neighbor] = proposed; previous[neighbor] = current;
                    queue.push(Item{proposed+heuristic(neighbor),counter++,neighbor});
                }
            }
        }
        return finish(5);
    } catch (...) { return finish(-3); }
}
