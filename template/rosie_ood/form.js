/*
 * Adapted from code written by Krishna Muriki 10/20
 */
'use strict'

/**
 *   Toggle the visibilty of a form group
 *
 *   @param      {string}    form_id  The form identifier
 *   @param      {boolean}   show     Whether to show or hide
 */
function toggle_visibility_of_form_group(form_id, show) {
  let form_element = $(form_id);
  let parent = form_element.parent();

  if(show) {
    parent.show();
  } else {
    form_element.val('');
    parent.hide();
  }
}

/**
 *  Toggle the visibilty of the indicated field
 */
function toggle_field_visibility(test_field_name, show_if_value, target_field_name) {
    let tested_field = $(test_field_name);
    toggle_visibility_of_form_group(
	target_field_name,
	tested_field.val() === show_if_value
    );
}

function toggle_type_of_use_field_visibility(){
    toggle_field_visibility(
	"#batch_connect_session_context_var_type_of_use",
	"Class",
	"#batch_connect_session_context_var_course_code");
    toggle_field_visibility(
	"#batch_connect_session_context_var_type_of_use",
	"Class",
	"#batch_connect_session_context_var_lab_number");
}

/**
 * Sets the change handler for the batch type of usage select.
 */
function set_type_ofuse_change_handler() {
  let type_ofuse = $("#batch_connect_session_context_var_type_of_use");
  type_ofuse.change(toggle_type_of_use_field_visibility);
}

/**
 *  Install event handlers
 */
$(document).ready(function() {
  // Ensure that fields are shown or hidden based on what was set in the last session
  toggle_type_of_use_field_visibility();

  set_type_ofuse_change_handler();
});

