/* MovieLens 推荐演示 —— 纯前端逻辑，数据来自 data/demo_data.js（离线预计算）。 */
(function () {
  "use strict";

  var D = window.DEMO_DATA;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif">' +
      '缺少 data/demo_data.js。请运行 python webapp/tools/export_demo_data.py 生成后再打开页面。</p>';
    return;
  }

  var GENRE_ZH = {
    "Action": "动作", "Adventure": "冒险", "Animation": "动画", "Children's": "儿童",
    "Comedy": "喜剧", "Crime": "犯罪", "Documentary": "纪录片", "Drama": "剧情",
    "Fantasy": "奇幻", "Film-Noir": "黑色电影", "Horror": "恐怖", "Musical": "歌舞",
    "Mystery": "悬疑", "Romance": "爱情", "Sci-Fi": "科幻", "Thriller": "惊悚",
    "War": "战争", "Western": "西部"
  };
  var AGE_ZH = { 1: "18 岁以下", 18: "18–24", 25: "25–34", 35: "35–44", 45: "45–49", 50: "50–55", 56: "56 岁以上" };
  var OCCUPATION_ZH = [
    "其他", "学术/教育", "艺术家", "文书/行政", "大学生", "客服", "医生/医护", "管理层/高管",
    "农民", "主妇", "K-12 学生", "律师", "程序员", "退休", "销售/市场", "科学家",
    "自由职业", "技师/工程师", "推销员", "待业", "作家"
  ];

  // ---------- 索引 ----------
  var movieById = new Map();
  D.movies.forEach(function (row) {
    movieById.set(row[0], {
      id: row[0], title: row[1], genres: String(row[2]).split("|"),
      year: row[3], n: row[4], npos: row[5], mean: row[6], bi: row[7]
    });
  });

  var userById = new Map();
  D.users.forEach(function (row) {
    userById.set(row[0], {
      id: row[0], gender: row[1], age: row[2], occupation: row[3],
      n: row[4], npos: row[5], mean: row[6], topGenres: row[7], likes: row[8]
    });
  });

  var totalN = 0, totalPos = 0;
  D.movies.forEach(function (row) { totalN += row[4]; totalPos += row[5]; });
  var prior = totalN ? totalPos / totalN : 0;

  // ---------- 工具 ----------
  function esc(text) {
    return String(text).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function $(id) { return document.getElementById(id); }

  // movies.dat 的标题自带年份，如 "Rear Window (1954)"，展示时拆开以免重复。
  function cleanTitle(title) {
    return String(title).replace(/\s*\(\d{4}\)\s*$/, "");
  }

  function genreTags(genres, limit) {
    var shown = limit ? genres.slice(0, limit) : genres;
    return shown.map(function (g) {
      return '<span class="gtag">' + esc(GENRE_ZH[g] || g) + "</span>";
    }).join("");
  }

  function pct(value) { return (value * 100).toFixed(1) + "%"; }

  // ---------- 顶部协议信息 ----------
  function renderProtocol() {
    var m = D.meta;
    var chips = [
      "评分 1,000,209 条",
      "用户 " + m.n_users.toLocaleString() + " 位",
      "电影 " + m.n_movies.toLocaleString() + " 部",
      "正例阈值 ≥ " + m.threshold,
      "随机种子 " + m.seed,
      "Top-" + m.top_k + " 推荐",
      "测试用户 " + m.n_test_users.toLocaleString() + " 位"
    ];
    $("protocolChips").innerHTML = chips.map(function (text) {
      return '<span class="chip">' + esc(text) + "</span>";
    }).join("");
  }

  // ---------- 推荐卡片 ----------
  function recCard(options) {
    var list = options.list;
    var body;
    if (!list || !list.length) {
      body = '<div class="empty">' + esc(options.emptyText || "暂无结果") + "</div>";
    } else {
      var scores = list.map(function (item) { return item[1]; });
      var lo = Math.min.apply(null, scores), hi = Math.max.apply(null, scores);
      var span = hi - lo || 1;
      body = '<ol class="rank">' + list.map(function (item) {
        var movie = movieById.get(item[0]);
        if (!movie) { return ""; }
        var bar = Math.round(8 + 92 * (item[1] - lo) / span);
        return '<li><div><div class="title">' + esc(cleanTitle(movie.title)) +
          (movie.year ? ' <span class="year">(' + movie.year + ")</span>" : "") + "</div>" +
          '<div class="genres">' + genreTags(movie.genres, 3) + "</div>" +
          '<div class="bar"><span style="width:' + bar + '%"></span></div></div>' +
          '<div class="score">' + item[1].toFixed(3) + "</div></li>";
      }).join("") + "</ol>";
    }
    return '<article class="rec-card">' +
      "<header><h3>" + esc(options.title) + '</h3><span class="badge ' + options.badgeClass + '">' +
      esc(options.badge) + "</span></header>" +
      '<p class="sub">' + esc(options.subtitle) + "</p>" + body +
      (options.note ? '<div class="rec-note">' + options.note + "</div>" : "") +
      "</article>";
  }

  function infoCard(options) {
    return '<article class="rec-card">' +
      "<header><h3>" + esc(options.title) + '</h3><span class="badge ' + options.badgeClass + '">' +
      esc(options.badge) + "</span></header>" +
      '<p class="sub">' + esc(options.subtitle) + "</p>" +
      '<ul class="notes">' + options.points.map(function (text) {
        return "<li>" + text + "</li>";
      }).join("") + "</ul></article>";
  }

  function rankNote(algo, uid) {
    var table = D.eval_ranks[algo];
    if (!table) { return ""; }
    var rank = table[String(uid)];
    if (rank === undefined) {
      return "该用户没有可评估的留出正例（正例不足 2 条），不在评估队列内。";
    }
    var hit = rank <= D.meta.top_k;
    return "留出正例（leave-one-out test 目标）在候选集全排序中的名次：<b>" + rank +
      "</b>，" + (hit ? '<b style="color:var(--ok)">命中 Top-' + D.meta.top_k + "</b>" : "未进入 Top-" + D.meta.top_k);
  }

  // ---------- 用户面板 ----------
  function renderProfile(user) {
    var liked = user.likes.map(function (id) {
      var movie = movieById.get(id);
      return movie ? "<li>" + esc(cleanTitle(movie.title)) + " <span class=\"gtag\">" +
        esc(GENRE_ZH[movie.genres[0]] || movie.genres[0]) + "</span></li>" : "";
    }).join("");
    var genres = user.topGenres.map(function (index) {
      return '<span class="facet">' + esc(GENRE_ZH[D.genres[index]] || D.genres[index]) + "</span>";
    }).join("");
    return '<div><h2 class="who">用户 #' + user.id + "</h2>" +
      '<div class="facets"><span class="facet">' + esc(user.gender === "F" ? "女" : "男") + "</span>" +
      '<span class="facet">' + esc(AGE_ZH[user.age] || ("年龄码 " + user.age)) + "</span>" +
      '<span class="facet">' + esc(OCCUPATION_ZH[user.occupation] || ("职业码 " + user.occupation)) + "</span>" +
      genres + "</div>" +
      '<dl class="stat-row"><div><dt>训练集评分</dt><dd>' + user.n + "</dd></div>" +
      "<div><dt>其中好评</dt><dd>" + user.npos + "</dd></div>" +
      "<div><dt>平均分</dt><dd>" + (user.mean === null ? "—" : user.mean.toFixed(2)) + "</dd></div>" +
      "<div><dt>好评率</dt><dd>" + (user.n ? pct(user.npos / user.n) : "—") + "</dd></div></dl></div>" +
      '<div><p class="mini-title">训练集中较新的好评（离线选取）</p><ul class="liked">' +
      (liked || "<li>无</li>") + "</ul></div>";
  }

  function overlapCard(uid) {
    var key = String(uid);
    var mf = (D.recs.mf[key] || []).map(function (x) { return x[0]; });
    var bpr = (D.recs.bpr[key] || []).map(function (x) { return x[0]; });
    var lrurec = (D.recs.lrurec[key] || []).map(function (x) { return x[0]; });

    function common(a, b) {
      return a.filter(function (x) { return b.indexOf(x) >= 0; });
    }
    var parts = ["<h2>Top-" + D.meta.top_k + " 推荐重合度</h2>"];
    var mfBpr = common(mf, bpr);
    parts.push("<p>MF 与 BPR 重合 <b>" + mfBpr.length + "/" + D.meta.top_k + "</b> 部：" +
      (mfBpr.length ? esc(mfBpr.map(function (id) { return cleanTitle(movieById.get(id).title); }).join("、")) : "无") + "</p>");
    if (lrurec.length) {
      var mfLr = common(mf, lrurec), bprLr = common(bpr, lrurec);
      parts.push("<p>MF 与 LRURec 重合 <b>" + mfLr.length + "/" + lrurec.length + "</b> 部，" +
        "BPR 与 LRURec 重合 <b>" + bprLr.length + "/" + lrurec.length + "</b> 部。</p>");
      parts.push('<p class="hint">该用户属于 LRURec 的 5 位演示用户，因此三者可直接对照；' +
        "其余用户只有 MF 与 BPR（LRURec 训练需要 GPU，结果已随报告归档）。</p>");
    } else {
      parts.push('<p class="hint">LRURec 仅内置 5 位示例用户（UserID 1–5）的离线结果，' +
        "本页对其余用户只展示 MF 与 BPR。三者使用同一候选集（3,706 部）与同一测试用户群（6,035 位），重合度可直接比较。</p>");
    }
    return parts.join("");
  }

  function showUser(rawId) {
    var uid = parseInt(rawId, 10);
    var hint = $("userHint");
    if (!isFinite(uid) || !userById.has(uid)) {
      hint.textContent = "请输入 1 – " + D.meta.n_users + " 之间的 MovieLens UserID。";
      hint.style.color = "var(--warn)";
      return;
    }
    hint.textContent = "已展示用户 #" + uid + " 的画像与三种算法的 Top-" + D.meta.top_k + "（已过滤该用户训练/验证/测试中观察过的电影）。";
    hint.style.color = "";
    $("userInput").value = uid;
    writeHash({ user: uid });

    var user = userById.get(uid);
    $("profileCard").innerHTML = renderProfile(user);
    $("profileCard").hidden = false;

    var key = String(uid);
    var lrurec = D.recs.lrurec[key];
    $("recsGrid").innerHTML = [
      recCard({
        title: "MF · 矩阵分解", badge: "课程方法", badgeClass: "mf",
        subtitle: "显式偏置 MF + ALS，候选集 " + D.meta.mf_catalogue.toLocaleString() + " 部，预测评分降序",
        list: D.recs.mf[key], note: rankNote("mf", uid)
      }),
      recCard({
        title: "BPR · 贝叶斯个性化排序", badge: "参考算法", badgeClass: "bpr",
        subtitle: "成对排序损失，候选集 " + D.meta.bpr_catalogue.toLocaleString() + " 部（与 MF 同口径），内积降序",
        list: D.recs.bpr[key],
        note: rankNote("bpr", uid) + " 候选集、划分与测试用户群均与 MF / LRURec 一致，指标可直接对比；但此处分数是<b>内积</b>而非评分，<b>不可与 MF 的预测评分直接比大小</b>。"
      }),
      recCard({
        title: "LRURec · 线性循环单元", badge: "2024 新方法", badgeClass: "lrurec",
        subtitle: "序列推荐，取用户最近 200 次行为作为输入",
        list: lrurec,
        emptyText: "该用户没有内置的 LRURec 结果。可点击上方示例用户 1–5 查看。",
        note: lrurec
          ? rankNote("lrurec", uid) + " LRURec 在官方仓库上训练需要 GPU，本页结果取自报告的离线运行记录。"
          : "LRURec 训练需要 CUDA GPU，因此本页只内置官方记录中的 5 位示例用户（UserID 1–5）。"
      })
    ].join("");

    $("overlapCard").innerHTML = overlapCard(uid);
    $("overlapCard").hidden = false;
  }

  // ---------- 示例档案（冷启动） ----------
  function renderGenres() {
    $("genreGrid").innerHTML = D.genres.map(function (g, index) {
      return '<label><input type="checkbox" value="' + esc(g) + '"' +
        (index < 2 ? " checked" : "") + "><span>" + esc(GENRE_ZH[g] || g) + "</span></label>";
    }).join("");
  }

  var SMOOTHING = 20; // 贝叶斯平滑强度：向全局好评率收缩

  function contentScore(movie) {
    return (movie.npos + SMOOTHING * prior) / (movie.n + SMOOTHING);
  }

  function buildProfile() {
    var picked = Array.prototype.slice
      .call(document.querySelectorAll("#genreGrid input:checked"))
      .map(function (input) { return input.value; });
    var hint = $("profileHint");
    if (!picked.length) {
      hint.textContent = "请至少选择 1 个类型。";
      hint.style.color = "var(--warn)";
      $("profileGrid").innerHTML = "";
      return;
    }
    hint.style.color = "";
    var minRating = parseFloat($("minRating").value);
    var minCount = parseInt($("minCount").value, 10);

    var pool = D.movies.filter(function (row) {
      if (row[4] < minCount) { return false; }
      if (minRating > 0 && (row[6] === null || row[6] < minRating)) { return false; }
      var genres = String(row[2]).split("|");
      return genres.some(function (g) { return picked.indexOf(g) >= 0; });
    }).map(function (row) {
      var movie = movieById.get(row[0]);
      return [row[0], Math.round(contentScore(movie) * 1000) / 1000];
    }).sort(function (a, b) { return b[1] - a[1] || a[0] - b[0]; }).slice(0, D.meta.top_k);

    var biasPool = D.movies.filter(function (row) { return row[7] !== null; })
      .map(function (row) { return [row[0], row[7]]; })
      .sort(function (a, b) { return b[1] - a[1] || a[0] - b[0]; }).slice(0, D.meta.top_k);

    hint.innerHTML = "已选择类型：" + picked.map(function (g) { return esc(GENRE_ZH[g] || g); }).join("、") +
      "；命中 " + pool.length + " 部候选（低于所选阈值或评分数不足的影片已剔除）。";

    $("profileGrid").innerHTML = [
      recCard({
        title: "内容 + 热度基线", badge: "冷启动", badgeClass: "mf",
        subtitle: "贝叶斯平滑好评率 =(好评数 + " + SMOOTHING + "×全局好评率)/(评分数 + " + SMOOTHING + ")",
        list: pool,
        note: "分数是<b>被该档案偏好的概率估计</b>，不是 MF/BPR 的预测评分，量纲不同，只用于本列内部排序。"
      }),
      recCard({
        title: "MF 冷启动回退 μ + bᵢ", badge: "非个性化", badgeClass: "bpr",
        subtitle: "新用户没有隐向量，MF 只能退化为「全局均分 + 电影偏置」",
        list: biasPool,
        note: "这就是协同过滤在冷启动下的真实表现：<b>对所有人给出同一张榜单</b>，与档案里选的类型无关。"
      }),
      infoCard({
        title: "两种策略的差别", badge: "为什么", badgeClass: "lrurec",
        subtitle: "左边用到内容属性，中间只用群体统计",
        points: [
          "左列利用电影的类型与评分统计，属于<b>内容/热度侧</b>策略：换一组类型偏好，榜单立刻不同。",
          "中列是 MF 在新用户上的真实输出：只有全局均分 μ 和电影偏置 bᵢ 可用，<b>与档案无关</b>。",
          "所以冷启动阶段只能给出「大致合理」的列表；用户产生若干真实评分后，才能切换到 MF/BPR 的个性化排序。",
          "工程上常见做法：内容/热度负责召回，协同模型负责排序，随交互数据积累逐步提高个性化权重。"
        ]
      })
    ].join("");
  }

  // ---------- 指标面板 ----------
  function renderMetrics() {
    var order = ["mf", "bpr", "lrurec"];
    var names = { mf: "MF（课程方法）", bpr: "BPR（参考算法）", lrurec: "LRURec（2024）" };
    var head = "<thead><tr><th>算法</th><th>候选集</th><th>测试用户</th><th>Recall@5</th><th>NDCG@5</th>" +
      "<th>Recall@10</th><th>NDCG@10</th><th>MRR@10</th></tr></thead><tbody>";
    var best = { "recall@10": -1, "ndcg@10": -1, "recall@5": -1, "ndcg@5": -1, "mrr@10": -1 };
    order.forEach(function (key) {
      var m = D.metrics[key] || {};
      Object.keys(best).forEach(function (metric) {
        if (m[metric] !== undefined && m[metric] > best[metric]) { best[metric] = m[metric]; }
      });
    });
    function cell(m, metric, digits) {
      if (m[metric] === undefined) { return '<td class="na">—</td>'; }
      var isBest = m[metric] === best[metric] ? ' class="best"' : "";
      return "<td" + isBest + ">" + m[metric].toFixed(digits === undefined ? 4 : digits) + "</td>";
    }
    var rows = order.map(function (key) {
      var m = D.metrics[key] || {};
      var catalogue = key === "bpr" ? D.meta.bpr_catalogue : D.meta.mf_catalogue;
      var users = m.users !== undefined ? m.users : D.meta.n_test_users;
      return "<tr><td>" + esc(names[key]) + "</td><td>" + catalogue.toLocaleString() + "</td><td>" +
        users.toLocaleString() + "</td>" + cell(m, "recall@5") + cell(m, "ndcg@5") +
        cell(m, "recall@10") + cell(m, "ndcg@10") + cell(m, "mrr@10") + "</tr>";
    }).join("");
    $("metricsTable").innerHTML = head + rows + "</tbody>";

    $("metricsNotes").innerHTML = [
      "三种算法在同一划分、同一候选集（3,706 部）与同一测试用户群（6,035 位）上评估，指标可直接横向比较；BPR 采用报告中的<b>同口径</b>结果（独立口径下为 6,038 位用户 / 3,883 部候选，数值差异小于 0.05%）。",
      "LRURec 的 NDCG@10 约是 MF 的 5.8 倍、BPR 的 2.8 倍：序列模型能利用「用户最近看过什么」的时序信息，而 MF/BPR 只把历史当作无序集合。",
      "页面展示的名次均已过滤用户已观察电影；离线指标同样如此，二者口径一致。"
    ].map(function (text) { return "<li>" + text + "</li>"; }).join("");
  }

  // ---------- 交互绑定 ----------
  function readHash() {
    var state = {};
    location.hash.replace(/^#/, "").split("&").forEach(function (pair) {
      var parts = pair.split("=");
      if (parts[0]) { state[parts[0]] = parts[1]; }
    });
    return state;
  }

  function writeHash(patch) {
    var state = readHash();
    Object.keys(patch).forEach(function (key) { state[key] = patch[key]; });
    var text = Object.keys(state).map(function (key) { return key + "=" + state[key]; }).join("&");
    if (location.hash.replace(/^#/, "") !== text) { location.hash = text; }
  }

  function activateTab(panelId) {
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
      tab.classList.toggle("is-active", tab.getAttribute("data-panel") === panelId);
    });
    Array.prototype.forEach.call(document.querySelectorAll(".panel"), function (panel) {
      panel.classList.toggle("is-active", panel.id === panelId);
    });
  }

  function bind() {
    $("maxUser").textContent = D.meta.n_users;
    $("userInput").max = D.meta.n_users;

    $("btnShow").addEventListener("click", function () { showUser($("userInput").value); });
    $("userInput").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { showUser($("userInput").value); }
    });
    $("btnRandom").addEventListener("click", function () {
      var ids = Array.from(userById.keys());
      showUser(ids[Math.floor(Math.random() * ids.length)]);
    });

    var quick = [1, 42, 4169, 6040].filter(function (id) { return userById.has(id); });
    $("quickUsers").innerHTML = quick.map(function (id) {
      return '<button data-user="' + id + '">示例用户 ' + id + "</button>";
    }).join("");
    $("quickUsers").addEventListener("click", function (event) {
      var id = event.target.getAttribute("data-user");
      if (id) { showUser(id); }
    });

    $("btnProfile").addEventListener("click", buildProfile);

    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
      tab.addEventListener("click", function () {
        var panelId = tab.getAttribute("data-panel");
        activateTab(panelId);
        writeHash({ tab: panelId });
      });
    });

    window.addEventListener("hashchange", function () { applyHash(); });
  }

  function applyHash() {
    var state = readHash();
    if (state.tab) { activateTab(state.tab); }
    if (state.user) { showUser(state.user); }
  }

  renderProtocol();
  renderGenres();
  renderMetrics();
  bind();
  buildProfile();

  var initial = readHash();
  if (initial.tab) { activateTab(initial.tab); }
  showUser(initial.user || 1);
})();
