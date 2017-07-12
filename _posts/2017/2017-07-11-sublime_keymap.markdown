---
layout: post
title:  "Sublime text 3 keymap"
date:   2017-07-11 9:00:00 +0100
categories: fast_copy_paste
---

{% highlight javascript %}
[
// Bookmarks
  { "keys": ["ctrl+space"], "command": "next_bookmark" }, 
  { "keys": ["ctrl+alt+space"], "command": "toggle_bookmark" },
  { "keys": ["ctrl+alt+shift+space"], "command": "clear_bookmarks" },
  { "keys": ["ctrl+alt+shift+."], "command": "select_all_bookmarks" },

// Autocomplete
{ "keys": ["alt+space"], "command": "auto_complete" },
{ "keys": ["alt+space"], "command": "replace_completion_with_auto_complete", "context":
  [
    { "key": "last_command", "operator": "equal", "operand": "insert_best_completion" },
    { "key": "auto_complete_visible", "operator": "equal", "operand": false },
    { "key": "setting.tab_completion", "operator": "equal", "operand": true }
  ]
},

// Cursor movements (heavy altgr usage)
  { "keys": ["ctrl+alt+q"], "command": "move_to", "args": {"to": "hardbol"} },
  { "keys": ["ctrl+alt+e"], "command": "move_to", "args": {"to": "hardeol"} },
  { "keys": ["ctrl+alt+shift+a"], "command": "move", "args": {"by": "words", "forward": false} },
  { "keys": ["ctrl+alt+shift+d"], "command": "move", "args": {"by": "word_ends", "forward": true} },
  { "keys": ["ctrl+alt+w"], "command": "move", "args": {"by": "lines", "forward": false} },
  { "keys": ["ctrl+alt+s"], "command": "move", "args": {"by": "lines", "forward": true} },
  { "keys": ["ctrl+alt+a"], "command": "move", "args": {"by": "characters", "forward": false} },
  { "keys": ["ctrl+alt+d"], "command": "move", "args": {"by": "characters", "forward": true} },

// Multiple selection
  { "keys": ["ctrl+alt+shift+w"], "command": "select_lines", "args": {"forward": false} },
  { "keys": ["ctrl+alt+shift+s"], "command": "select_lines", "args": {"forward": true} },

// Comments
  { "keys": ["ctrl+,"], "command": "toggle_comment", "args": { "block": false } },
  { "keys": ["ctrl+shift+,"], "command": "toggle_comment", "args": { "block": true } },


// Incremental search
    { "keys": ["ctrl+i"], "command": "show_panel", 
           "args": {"panel": "find", "reverse":false} },
       { "keys": ["ctrl+shift+i"], "command": "show_panel", 
           "args": {"panel": "find", "reverse":true} },
       { "keys": ["ctrl+i"], "command": "find_next",
           "context":
           [
               {"key": "panel", "operand": "find"},
               { "key": "panel_visible", "operator": "equal", "operand": true }
           ]
       },
       { "keys": ["ctrl+shift+i"], "command": "find_prev",
           "context":
           [
               {"key": "panel", "operand": "find"},
               { "key": "panel_visible", "operator": "equal", "operand": true }
           ]
       },
       { "keys": ["enter"], "command": "hide_after_word",
           "context":
           [
               {"key": "panel", "operand": "find"},
               { "key": "panel_visible", "operator": "equal", "operand": true }
           ]
       },
       {
           "keys": ["f5"],
           "command": "revert"
       },

// Console
  { "keys": ["ctrl+0"], "command": "show_panel", "args": {"panel": "console", "toggle": true} },


// Altgr+p = escape
  { "keys": ["ctrl+alt+p"], "command": "single_selection", "context":
    [
      { "key": "num_selections", "operator": "not_equal", "operand": 1 }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "clear_fields", "context":
    [
      { "key": "has_next_field", "operator": "equal", "operand": true }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "clear_fields", "context":
    [
      { "key": "has_prev_field", "operator": "equal", "operand": true }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "hide_panel", "args": {"cancel": true},
    "context":
    [
      { "key": "panel_visible", "operator": "equal", "operand": true }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "hide_overlay", "context":
    [
      { "key": "overlay_visible", "operator": "equal", "operand": true }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "hide_popup", "context":
    [
      { "key": "popup_visible", "operator": "equal", "operand": true }
    ]
  },
  { "keys": ["ctrl+alt+p"], "command": "hide_auto_complete", "context":
    [
      { "key": "auto_complete_visible", "operator": "equal", "operand": true }
    ]
  }
]
{% endhighlight %}
