/* nib-rs docs — tiny, dependency-free enhancements */
(function () {
  "use strict";

  var root = document.documentElement;
  var themeKey = "nib-docs-theme";
  var themeBtn = document.getElementById("themeToggle");
  var menuBtn = document.getElementById("menuToggle");

  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function currentTheme() {
    return root.getAttribute("data-theme") || (systemDark() ? "dark" : "light");
  }

  function paintThemeButton() {
    if (!themeBtn) return;
    var dark = currentTheme() === "dark";
    themeBtn.textContent = dark ? "☀" : "☾";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    themeBtn.title = themeBtn.getAttribute("aria-label");
  }

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    try { localStorage.setItem(themeKey, theme); } catch (e) {}
    paintThemeButton();
  }

  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      setTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }
  paintThemeButton();

  if (menuBtn) {
    menuBtn.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", function (event) {
      if (!document.body.classList.contains("nav-open")) return;
      var sidebar = document.getElementById("sidebar");
      if (sidebar && !sidebar.contains(event.target) && !menuBtn.contains(event.target)) {
        document.body.classList.remove("nav-open");
        menuBtn.setAttribute("aria-expanded", "false");
      }
    });
  }

  // Active page in the sidebar.
  var here = (location.pathname.split("/").pop() || "index.html").toLowerCase();
  document.querySelectorAll(".sidebar nav a[href]").forEach(function (link) {
    var href = (link.getAttribute("href") || "").toLowerCase();
    if (href === here) {
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
    }
  });

  // Heading anchors + table of contents.
  var article = document.querySelector("article");
  var toc = document.getElementById("toc");
  if (article) {
    var used = {};
    var headings = article.querySelectorAll("h2, h3");
    headings.forEach(function (heading, index) {
      if (!heading.id) {
        var base = (heading.textContent || "section")
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "")
          .slice(0, 60) || "section";
        var id = base;
        var n = 2;
        while (used[id] || document.getElementById(id)) { id = base + "-" + n++; }
        heading.id = id;
      }
      used[heading.id] = true;

      var anchor = document.createElement("a");
      anchor.className = "anchor";
      anchor.href = "#" + heading.id;
      anchor.setAttribute("aria-label", "Link to this section");
      anchor.textContent = "#";
      heading.appendChild(anchor);

      if (toc && heading.tagName === "H2") {
        var item = document.createElement("a");
        item.href = "#" + heading.id;
        item.textContent = heading.childNodes[0].textContent.trim();
        item.dataset.target = heading.id;
        toc.appendChild(item);
      }
    });

    // Scrollspy for the generated TOC.
    if (toc && "IntersectionObserver" in window) {
      var links = Array.prototype.slice.call(toc.querySelectorAll("a"));
      var byId = {};
      links.forEach(function (link) { byId[link.dataset.target] = link; });
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (link) { link.classList.remove("active"); });
          var link = byId[entry.target.id];
          if (link) link.classList.add("active");
        });
      }, { rootMargin: "-80px 0px -70% 0px", threshold: 0 });
      article.querySelectorAll("h2").forEach(function (h) { observer.observe(h); });
    }
  }

  // Copy buttons on every code block.
  document.querySelectorAll("pre").forEach(function (pre) {
    var button = document.createElement("button");
    button.className = "copy";
    button.type = "button";
    button.textContent = "Copy";
    button.addEventListener("click", function () {
      var code = pre.querySelector("code");
      var text = code ? code.innerText : pre.innerText;
      function done() {
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = "Copy"; }, 1400);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
    pre.appendChild(button);
  });

  function fallback(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); done(); } catch (e) {}
    document.body.removeChild(area);
  }

  // Close the mobile drawer after choosing a link.
  document.querySelectorAll(".sidebar a[href]").forEach(function (link) {
    link.addEventListener("click", function () {
      document.body.classList.remove("nav-open");
      if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
    });
  });
})();
