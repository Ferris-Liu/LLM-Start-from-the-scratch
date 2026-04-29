window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],   // 行内公式：\( ... \)
    displayMath: [["\\[", "\\]"]],  // 块级公式：\[ ... \]
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"  // 只处理 arithmatex 标记的元素
  }
};

// 页面切换时（navigation.instant）重新渲染公式
document$.subscribe(() => {
  MathJax.typesetPromise()
})
