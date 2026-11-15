// Compact navigation below 1000 px: a menu button that opens the links as a list.
(function () {
  const nav = document.getElementById("nav");
  const links = nav && nav.querySelector(".links");
  if (!links) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "menu mono";
  links.id = links.id || "navlinks";
  btn.setAttribute("aria-controls", links.id);
  const set = open => {
    nav.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "close" : "menu";
  };
  set(false);
  nav.querySelector(".grid").append(btn);
  btn.addEventListener("click", () => set(!nav.classList.contains("open")));
  links.addEventListener("click", e => { if (e.target.closest("a")) set(false); });
  addEventListener("keydown", e => { if (e.key === "Escape") set(false); });
  addEventListener("resize", () => { if (innerWidth > 1000) set(false); });
})();
